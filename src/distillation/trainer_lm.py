import logging
import os

import mlflow
from torch.cuda import amp
from tqdm import trange

from coreset import CoresetModule
from data import DataModule
from evaluator import Evaluator
from generator import GeneratorModel
from learner import LearnerModel
from utils import average, batch_to_cuda, configure_optimizer, endless_dataloader

from .trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class TrainerLM(TrainerBase):
    def fit(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        data_module: DataModule,
        evaluator: Evaluator,
        repset_teachers: None,
        coreset_module: CoresetModule,
    ):
        generator.cuda()
        learner.cuda()

        train_loader = data_module.get_train_loader(
            batch_size=self.config.lm_batch_size
        )
        train_loader = endless_dataloader(
            train_loader, max_iteration=self.config.total_train_step
        )

        optimizer, scheduler = self.generator_optimizer(generator)
        scaler = amp.GradScaler(enabled=self.use_amp)

        best_ckpt_path = os.path.join(self.config.save_model_dir, "best-ckpt")
        train_logs = []
        best_val_score = float("-inf")
        logger.info("Start training!!")
        for it in trange(
            self.config.total_train_step,
            dynamic_ncols=True,
            leave=False,
            desc="Training generator",
        ):
            # evaluate before training
            if it % self.config.val_interval == 0 and it >= self.config.val_skip_step:
                results = self.evaluate(
                    generator, learner, evaluator, data_module, coreset_module, step=it
                )
                if results[f"valid.{evaluator.metric_key}"] > best_val_score:
                    best_val_score = results[f"valid.{evaluator.metric_key}"]
                    generator.save_model(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

            generator.train()
            batch_real = next(train_loader)

            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                loss_lm = generator.compute_loss(
                    **batch_to_cuda(batch_real["generator"])
                )
                loss_lm = loss_lm.mean()

            # compute gradient
            scaler.scale(loss_lm).backward()
            self.train_step(generator, optimizer, scheduler, scaler)
            train_log = {"train.loss_lm": loss_lm.item()}

            train_logs.append(train_log)

            if (it + 1) % self.config.log_interval == 0:
                train_logs = average(train_logs)
                train_logs["train.lr"] = scheduler.get_last_lr()[0]
                mlflow.log_metrics(train_logs, step=it + 1)
                logger.info(
                    "TRAIN [{:>{}}/{}]: {}".format(
                        it + 1,
                        len(str(self.config.total_train_step)),
                        self.config.total_train_step,
                        train_logs,
                    )
                )
                train_logs = []

        results = self.evaluate(
            generator, learner, evaluator, data_module, coreset_module, step=it + 1
        )

        if results[f"valid.{evaluator.metric_key}"] > best_val_score:
            best_val_score = results[f"valid.{evaluator.metric_key}"]
            generator.save_model(best_ckpt_path)
            logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_model_dir, "last-ckpt")
        generator.save_model(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # save tokenizer
        tokenizer_path = os.path.join(self.config.save_model_dir, "tokenizer")
        generator.save_tokenizer(tokenizer_path)

        # load best checkpoint
        generator.load_model(best_ckpt_path)

        logger.info("Finish training!!")

    def generator_optimizer(self, generator: GeneratorModel):
        return configure_optimizer(
            generator,
            lr=self.config.lr,
            optimizer_type=self.config.optimizer_type,
            scheduler_type=self.config.scheduler_type,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            num_train_steps=self.config.total_train_step,
        )
