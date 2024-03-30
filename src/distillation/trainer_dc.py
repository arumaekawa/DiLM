import logging
import os
from typing import Generator

import mlflow
import torch
from datasets import Dataset, concatenate_datasets
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from coreset import CoresetModule
from data import DataModule
from evaluator import EvaluateConfig, Evaluator
from generator import GeneratorModel
from learner import LearnerModel
from utils import average, batch_to_cuda, configure_optimizer, endless_dataloader

from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class TrainerDC(TrainerBase):
    def fit(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        data_module: DataModule,
        evaluator: Evaluator,
        repset_teachers: list[Dataset] | None,
        coreset_module: CoresetModule,
    ):
        generator.cuda()
        learner.cuda()

        num_labels = data_module.num_labels

        assert self.config.total_train_step % self.config.inner_loop == 0
        outer_loop = self.config.total_train_step // self.config.inner_loop
        assert self.config.val_interval % self.config.inner_loop == 0
        assert outer_loop % (self.config.val_interval // self.config.inner_loop) == 0
        assert self.config.log_interval % self.config.inner_loop == 0
        assert outer_loop % (self.config.log_interval // self.config.inner_loop) == 0

        # setup data loader for gm loss
        gm_real_loaders = self.get_gm_real_loaders(
            data_module, learner=learner, repset_teachers=repset_teachers
        )

        # setup data loader for lm loss
        if self.config.lm_lambda > 0:
            lm_loader = self.get_lm_loader(data_module)

        # setup data loader for updating learner
        if self.config.inner_loop > 1:
            learner_train_loader = self.get_learner_train_loader(data_module)

        if not self.config.use_generated_data:
            raise NotImplementedError
            # if self.config.n_clusters_for_syn_sampler > 1:
            #     gm_syn_loaders = self.cluster_wise_dataloader(
            #         data_module.preprocessed_datasets["train"],
            #         data_module=data_module,
            #         learner=learner,
            #         dpc=self.config.gm_syn_dpc,
            #         n_clusters=self.config.n_clusters_for_syn_sampler,
            #         max_iteration=self.config.inner_loop
            #         * self.config.generate_dataset_interval,
            #     )
            # else:
            #     gm_syn_loaders = {
            #         label: endless_dataloader(
            #             data_module.get_train_loader(
            #                 batch_size=self.config.gm_syn_dpc,
            #                 shuffle=False,
            #                 drop_last=False,
            #                 label=label,
            #             ),
            #             max_iteration=self.config.total_train_step,
            #         )
            #         for label in range(num_labels)
            #     }

        # setup optimizer
        optimizer, scheduler = self.generator_optimizer(generator)
        scaler = amp.GradScaler(enabled=self.use_amp)
        optimizer.zero_grad(set_to_none=True)

        # setup for torch.functional_call
        params = {k: v.detach() for k, v in learner.named_parameters()}
        buffers = {k: v for k, v in learner.named_buffers()}

        # save tokenizer
        tokenizer_path = os.path.join(self.config.save_model_dir, "tokenizer")
        generator.save_tokenizer(tokenizer_path)

        # save best model path
        best_ckpt_path = os.path.join(self.config.save_model_dir, "best-ckpt")

        train_logs = []
        best_val_score = float("-inf")
        logger.info("Start training!!")
        for ol in trange(
            outer_loop, dynamic_ncols=True, leave=False, desc="Outer Loop"
        ):
            # train_step = ol * self.config.inner_loop
            # evaluate before training
            if (
                (ol * self.config.inner_loop) % self.config.val_interval == 0
                and ol * self.config.inner_loop >= self.config.val_skip_step
            ):
                results = self.evaluate(
                    generator,
                    learner,
                    evaluator,
                    data_module,
                    coreset_module,
                    step=ol * self.config.inner_loop,
                )
                if results[f"valid.{evaluator.metric_key}"] > best_val_score:
                    best_val_score = results[f"valid.{evaluator.metric_key}"]
                    generator.save_model(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

            # generate dataset
            if (
                ol % self.config.generate_dataset_interval == 0
                and self.config.use_generated_data
            ):
                logger.info(
                    "TRAIN [{:>{}}/{}]: Generate synthetic data".format(
                        ol * self.config.inner_loop,
                        len(str(self.config.total_train_step)),
                        self.config.total_train_step,
                    )
                )
                gm_syn_loaders = self.get_gm_syn_loaders(
                    generator, learner, data_module
                )

            learner.init_weights()
            if self.config.inner_loop > 1:
                learner_optimizer, learner_scheduler = self.learner_optimizer(
                    learner, evaluate_config=evaluator.config
                )
                learner_scaler = amp.GradScaler(enabled=self.use_amp)

            generator.train()

            outer_loop_train_logs = []
            for outer_step in trange(
                self.config.inner_loop,
                dynamic_ncols=True,
                leave=False,
                desc="Inner loop",
            ):
                # compute DC loss
                grad_sim = 0.0
                if self.config.lm_lambda < 1:
                    for label in range(num_labels):
                        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            # compute gradient with real samples
                            # sample size: gm_real_dpc * gm_real_grad_accum_step
                            with torch.no_grad():
                                grad_real_list = []
                                for _ in range(self.config.gm_real_grad_accum_step):
                                    batch_gm_real = next(gm_real_loaders[label])
                                    grad_real = self.compute_grad(
                                        learner=learner,
                                        params=params,
                                        buffers=buffers,
                                        **batch_to_cuda(batch_gm_real["learner"]),
                                    )
                                    grad_real_list.append(grad_real)

                                grad_real = torch.stack(grad_real_list).mean(0)

                            # compute generation probability
                            batch_gm_syn = next(gm_syn_loaders[label])
                            gen_losses = generator.compute_loss(
                                **batch_to_cuda(batch_gm_syn["generator"])
                            )
                            loss_weights = F.softmax(
                                -gen_losses / self.config.normalize_temperature,
                                dim=-1,
                            )
                            # compute gradient with loss weights
                            grad_syn = self.compute_grad(
                                learner=learner,
                                params=params,
                                buffers=buffers,
                                **batch_to_cuda(batch_gm_syn["learner"]),
                                loss_weights=loss_weights,
                            )
                            grad_sim_label = F.cosine_similarity(
                                grad_real, grad_syn, dim=0
                            )
                            loss_dc_label = (1 - grad_sim_label) / num_labels

                        # backward for each label
                        scaler.scale(
                            loss_dc_label * (1 - self.config.lm_lambda)
                        ).backward()

                        grad_sim += grad_sim_label.item()

                    grad_sim /= num_labels
                    loss_dc = 1 - grad_sim
                else:
                    loss_dc = 0.0

                if self.config.lm_lambda > 0:
                    batch_lm = next(lm_loader)
                    with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss_lm = generator.compute_loss(
                            **batch_to_cuda(batch_lm["generator"])
                        )
                        loss_lm = loss_lm.mean()

                    # backward for lm loss
                    scaler.scale(loss_lm * self.config.lm_lambda).backward()
                    loss_lm = loss_lm.item()
                else:
                    loss_lm = 0.0

                loss = (
                    loss_dc * (1 - self.config.lm_lambda)
                    + loss_lm * self.config.lm_lambda
                )

                # update generator
                self.train_step(generator, optimizer, scheduler, scaler)

                outer_loop_train_log = {
                    "train.loss": loss,
                    "train.loss_dc": loss_dc,
                    "train.loss_lm": loss_lm,
                    "train.grad_sim": grad_sim,
                }
                outer_loop_train_logs.append(outer_loop_train_log)

                # update learner
                if (outer_step + 1) < self.config.inner_loop:
                    for _ in range(self.config.model_step_per_inner_step):
                        batch_learner = next(learner_train_loader)
                        batch_learner = batch_to_cuda(batch_learner["learner"])
                        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            outputs = learner(**batch_to_cuda(batch_learner))
                            loss_learner = outputs.loss.mean()

                        learner_scaler.scale(loss_learner).backward()
                        self.train_step(
                            learner,
                            learner_optimizer,
                            learner_scheduler,
                            learner_scaler,
                        )

            average_outer_loop_train_logs = average(outer_loop_train_logs)
            train_logs.append(average_outer_loop_train_logs)

            if ((ol + 1) * self.config.inner_loop) % self.config.log_interval == 0:
                train_logs = average(train_logs)
                train_logs["train.lr"] = scheduler.get_last_lr()[0]

                mlflow.log_metrics(train_logs, step=(ol + 1) * self.config.inner_loop)
                logger.info(
                    "TRAIN [{:>{}}/{}]: {}".format(
                        (ol + 1) * self.config.inner_loop,
                        len(str(self.config.total_train_step)),
                        self.config.total_train_step,
                        train_logs,
                    )
                )
                train_logs = []

        logger.info("Finish training!!")

        results = self.evaluate(
            generator,
            learner,
            evaluator,
            data_module,
            coreset_module,
            step=(ol + 1) * self.config.inner_loop,
        )

        if results[f"valid.{evaluator.metric_key}"] > best_val_score:
            best_val_score = results[f"valid.{evaluator.metric_key}"]
            generator.save_model(best_ckpt_path)
            logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_model_dir, "last-ckpt")
        generator.save_model(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        generator.load_model(best_ckpt_path)

    def syn_data_to_batch(self, syn_data: Dataset, data_module: DataModule):
        syn_data = data_module.preprocess_dataset(syn_data)
        train_loader = DataLoader(
            syn_data,
            batch_size=len(syn_data),
            collate_fn=data_module.data_collator,
            shuffle=True,
            drop_last=True,
        )
        return next(iter(train_loader))

    def compute_grad(
        self,
        learner: LearnerModel,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        loss_weights: torch.Tensor | None = None,
        input_ids=torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return flatten gradient vector"""

        def compute_loss(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            input_ids: dict[str, torch.LongTensor],
            loss_weights: torch.Tensor | None = None,
            **kwargs,
        ):
            outputs = torch.func.functional_call(
                learner, (params, buffers), args=input_ids, kwargs=kwargs
            )
            loss = outputs.loss

            if loss_weights is None:
                return loss.mean()

            assert loss.shape == loss_weights.shape
            return loss.dot(loss_weights)

        grads = torch.func.grad(compute_loss)(
            params,
            buffers,
            input_ids=input_ids,
            loss_weights=loss_weights,
            **kwargs,
        )

        if self.config.classifier_grad_only:
            grads = {
                name: param
                for name, param in grads.items()
                if name in list(learner.classifier_param_names())
            }

        return torch.concat([grad.view(-1) for grad in grads.values()], dim=0)

    def learner_optimizer(self, learner: LearnerModel, evaluate_config: EvaluateConfig):
        return configure_optimizer(
            learner,
            lr=evaluate_config.lr,
            optimizer_type=evaluate_config.optimizer_type,
            scheduler_type=evaluate_config.scheduler_type,
            weight_decay=evaluate_config.weight_decay,
            warmup_ratio=evaluate_config.warmup_ratio,
            num_train_steps=self.config.inner_loop
            * self.config.model_step_per_inner_step,
        )

    def get_gm_real_loaders(
        self,
        data_module: DataModule,
        learner: LearnerModel | None = None,
        repset_teachers: list[Dataset] | None = None,
    ) -> dict[int, Generator[dict[str, torch.Tensor], None, None]]:
        """Return dataloader of real data for gradient matching for each label"""
        num_labels = data_module.num_labels

        # setup data loader for gm loss
        if self.config.repset_teacher:
            # use repset teachers
            assert repset_teachers is not None
            assert self.config.gm_real_grad_accum_step <= self.config.n_repset
            concat_repset_teachers = concatenate_datasets(repset_teachers)
            repset_size = self.config.repset_dpc * num_labels
            assert len(concat_repset_teachers) == repset_size * self.config.n_repset

            return {
                label: endless_dataloader(
                    data_module.get_train_loader(
                        dataset=concat_repset_teachers,
                        batch_size=self.config.gm_real_dpc,
                        label=label,
                        shuffle=False,
                        drop_last=False,
                    ),
                    max_iteration=self.config.total_train_step
                    * self.config.gm_real_grad_accum_step,
                )
                for label in range(num_labels)
            }

        else:
            # use real data
            assert not self.config.repset_teacher
            if self.config.n_clusters_for_real_sampler > 1:
                return self.cluster_wise_dataloader(
                    data_module.preprocessed_datasets["train"],
                    data_module=data_module,
                    learner=learner,
                    dpc=self.config.gm_real_dpc,
                    n_clusters=self.config.n_clusters_for_real_sampler,
                    max_iteration=self.config.inner_loop
                    * self.config.generate_dataset_interval,
                )
            else:
                return {
                    label: endless_dataloader(
                        data_module.get_train_loader(
                            batch_size=self.config.gm_real_dpc,
                            label=label,
                        ),
                        max_iteration=self.config.total_train_step
                        * self.config.gm_real_grad_accum_step,
                    )
                    for label in range(num_labels)
                }

    def get_lm_loader(self, data_module: DataModule) -> Generator[dict, None, None]:
        """Return dataloader of real data for language modeling loss"""
        lm_loader = data_module.get_train_loader(batch_size=self.config.lm_batch_size)
        lm_loader = endless_dataloader(
            lm_loader, max_iteration=self.config.total_train_step
        )
        return lm_loader

    def get_learner_train_loader(
        self, data_module: DataModule
    ) -> Generator[dict, None, None]:
        """Return dataloader of real data for language modeling loss"""
        learner_train_step = (
            self.config.total_train_step * self.config.model_step_per_inner_step
        )
        learner_train_loader = data_module.get_train_loader()
        learner_train_loader = endless_dataloader(
            learner_train_loader, max_iteration=learner_train_step
        )
        return learner_train_loader

    def get_gm_syn_loaders(
        self, generator: GeneratorModel, learner: LearnerModel, data_module: DataModule
    ) -> dict[int, Generator[dict[str, torch.Tensor], None, None]]:
        """Return dataloader of synthetic data for gradient matching for each label"""

        # generate synthetic data
        syn_datasets = generator.generate_dataset(
            dpc=self.config.gm_syn_dpc,
            n=self.config.inner_loop * self.config.generate_dataset_interval,
        )

        # preprocess synthetic data
        preprocessed_syn_dataset = data_module.preprocess_dataset(
            concatenate_datasets(syn_datasets)
        )

        # setup dataloader
        if self.config.n_clusters_for_syn_sampler > 1:
            # return cluster-wise balanced dataloader
            return self.cluster_wise_dataloader(
                preprocessed_syn_dataset,
                data_module=data_module,
                learner=learner,
                dpc=self.config.gm_syn_dpc,
                n_clusters=self.config.n_clusters_for_syn_sampler,
                max_iteration=self.config.inner_loop
                * self.config.generate_dataset_interval,
            )

        # return random dataloader
        gm_syn_loaders = {
            label: iter(
                data_module.get_train_loader(
                    dataset=preprocessed_syn_dataset,
                    batch_size=self.config.gm_syn_dpc,
                    shuffle=False,
                    drop_last=False,
                    label=label,
                )
            )
            for label in range(data_module.num_labels)
        }

        return gm_syn_loaders
