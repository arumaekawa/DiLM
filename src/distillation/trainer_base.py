import logging
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generator

import mlflow
import torch
from datasets import Dataset
from torch import nn
from torch.cuda import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from coreset import CoresetModule
from coreset.k_centers import FastKMeans
from data import DataModule
from evaluator import Evaluator
from generator import GeneratorModel
from learner import LearnerModel
from utils import average, batch_to_cuda, configure_optimizer

from .distilled_data import DistilledDataConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_type: str = "dc"  # lm, dc, mtt
    normalize_temperature: float = 1.0

    gm_syn_dpc: int = 50
    gm_real_dpc: int = 50
    gm_real_grad_accum_step: int = 1

    lm_lambda: float = 0.01
    lm_batch_size: int = 64

    n_clusters_for_real_sampler: int = 1
    n_clusters_for_syn_sampler: int = 1
    use_generated_data: bool = True

    classifier_grad_only: bool = False

    # repset teacher
    repset_teacher: bool = False
    repset_dpc: int = 50
    n_repset: int = 1

    total_train_step: int = 20000
    inner_loop: int = 5
    model_step_per_inner_step: int = 10

    generate_dataset_interval: int = 10

    optimizer_type: str = "adamw"  # ["sgd", "adam"]
    scheduler_type: str = "linear"
    lr: float = 1.0e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float | None = 2.0
    val_interval: int = 2000
    val_skip_step: int = 0
    log_interval: int = 100  # if -1 -> len(dataloader)//10
    save_model_dir: str = "path/to/checkpoint_dir"
    save_valid_result_dir: str = "path/to/validation_result_dir"
    fp16: bool = False
    bf16: bool = False


class TrainerBase(metaclass=ABCMeta):
    def __init__(self, config: TrainConfig, distilled_data_config: DistilledDataConfig):
        self.config = config
        self.distilled_data_config = distilled_data_config

    @abstractmethod
    def fit(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        data_module: DataModule,
        evaluator: Evaluator,
        coreset_module: CoresetModule,
    ):
        raise NotImplementedError

    def train_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: amp.GradScaler,
    ):
        assert scheduler.get_last_lr()[0] >= 0

        # gradient clipping
        if self.config.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.config.max_grad_norm,
            )

        # update distilled data
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        optimizer.zero_grad(set_to_none=True)

    def evaluate(
        self,
        generator: GeneratorModel,
        learner: LearnerModel,
        evaluator: Evaluator,
        data_module: DataModule,
        coreset_module: CoresetModule,
        step: int,
    ):
        generate_dpc = int(
            self.distilled_data_config.dpc
            * self.distilled_data_config.over_sample_ratio
        )
        dataset_list = generator.generate_dataset(dpc=generate_dpc, n=5)
        if self.distilled_data_config.over_sample_ratio > 1.0:
            dataset_list = [
                coreset_module.get_coreset(dataset, dpc=self.distilled_data_config.dpc)
                for dataset in dataset_list
            ]

        results = evaluator.evaluate(
            dataset_list=dataset_list,
            learner=learner,
            data_module=data_module,
            save_result_dir=os.path.join(
                self.config.save_valid_result_dir, f"step_{step}"
            ),
            verbose=True,
        )

        results = {f"valid.{k}": v for k, v in average(results).items()}
        logger.info(
            "Validation [{:>{}}/{}]: {}".format(
                step,
                len(str(self.config.total_train_step)),
                self.config.total_train_step,
                results,
            )
        )
        mlflow.log_metrics(results, step=step)

        return results

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

    def cluster_wise_dataloader(
        self,
        preprocessed_dataset: Dataset,
        data_module: DataModule,
        learner: LearnerModel,
        dpc: int,
        n_clusters: int,
        max_iteration: int = 1000,
    ) -> dict[int, Generator[dict[str, torch.Tensor], None, None]]:
        """
        Return cluster-wise balanced dataloader. Each batch produced by this dataloader
        contains the same number of data from each cluster.
        """

        assert "learner" in preprocessed_dataset.features
        assert "generator" in preprocessed_dataset.features

        assert dpc % n_clusters == 0

        def cluster_wise_dataloader(
            clustered_dataset_list: list[Dataset], batch_size: int
        ) -> Generator[dict[str, dict], None, None]:
            assert batch_size % len(clustered_dataset_list) == 0
            batch_size_per_cluster = batch_size // len(clustered_dataset_list)

            data_queue = []
            for i in range(batch_size_per_cluster * max_iteration):
                for cluster_dataset in clustered_dataset_list:
                    data_queue.append(cluster_dataset[i % len(cluster_dataset)])

            assert len(data_queue) == batch_size * max_iteration

            data_loader = data_module.get_train_loader(
                data_queue,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            for batch in data_loader:
                yield batch

        data_loaders = {}
        for label in range(data_module.num_labels):
            label_dataset = preprocessed_dataset.filter(
                lambda ex: ex["learner"]["labels"] == label
            )
            clustered_dataset = self.clustering_dataset(
                label_dataset, learner, data_module, n_clusters
            )
            # -> clustered_dataset has column named "cluster_id"
            assert "cluster_id" in clustered_dataset.features.keys()

            clustered_dataset_list = [
                clustered_dataset.filter(
                    lambda ex: ex["cluster_id"] == i
                ).remove_columns("cluster_id")
                for i in range(n_clusters)
            ]
            logger.info(
                "Clustered dataset: size={}".format(
                    sorted([len(dataset) for dataset in clustered_dataset_list])
                )
            )
            data_loaders[label] = cluster_wise_dataloader(
                clustered_dataset_list, batch_size=dpc
            )

        return data_loaders

    def clustering_dataset(
        self,
        dataset: Dataset,
        learner: LearnerModel,
        data_module: DataModule,
        n_clusters: int,
    ) -> Dataset:
        """
        Clustering dataset with learner model,
        and add column named "cluster_id" to dataset.
        """

        learner.cuda()
        learner.eval()

        def get_embedding(batch):
            batch = [
                {"learner": b_ln, "generator": b_gn}
                for b_ln, b_gn in zip(batch["learner"], batch["generator"])
            ]
            batch = data_module.data_collator(batch)
            with torch.inference_mode():
                outputs = learner(
                    **batch_to_cuda(batch["learner"]), output_hidden_states=True
                )
                embeddings = outputs.hidden_states[-1][:, 0].cpu()
            return {"embedding": embeddings}

        embed_dataset = dataset.map(get_embedding, batched=True, batch_size=512)
        embeddings = torch.tensor(embed_dataset["embedding"])

        logger.info("Clustering dataset")
        kmeans = FastKMeans(n_clusters=n_clusters)
        cluster_ids, _ = kmeans.fit_predict(embeddings)

        assert len(cluster_ids) == len(dataset)

        dataset = dataset.add_column("cluster_id", cluster_ids)

        return dataset

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
