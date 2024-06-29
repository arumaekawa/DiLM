import logging
import os
import warnings
from dataclasses import dataclass

from datasets import Dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer

from dataset_attrs import DATASET_ATTRS
from generator import GeneratorModel

from .herding import herding
from .k_centers import k_centers
from .random import random_selection
from .rank_dilm import rank_with_dilm

logger = logging.getLogger(__name__)


ENCODER_MODEL_REQUIRED_METHODS = {"k_centers", "herding"}


@dataclass
class CoresetConfig:
    coreset_type: str = "random"  # random, k_centers, dilm
    model_name: str = "bert-base-uncased"  # for embedding based method
    save_dir: str = "dataset/save/directory"


class CoresetModule:
    def __init__(
        self,
        config: CoresetConfig,
        task_name: str,
        dataset: Dataset,
        generator: GeneratorModel | None = None,
    ):
        self.config = config
        self.task_name = task_name
        self.num_labels = DATASET_ATTRS[task_name]["num_labels"]
        self.dataset = dataset

        assert config.coreset_type != "rank_dilm" or generator is not None
        self.generator = generator

        if config.coreset_type in ENCODER_MODEL_REQUIRED_METHODS:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.encoder = AutoModel.from_pretrained(config.model_name)
            assert "pad_token" in self.encoder_tokenizer.special_tokens_map.keys()

            self.save_datasets_dir = os.path.join(
                self.config.save_dir, self.config.model_name
            )
        else:
            self.save_datasets_dir = self.config.save_dir

    def generate_dataset(self, dpc: int, n: int = 1) -> list[Dataset]:

        os.makedirs(self.save_datasets_dir, exist_ok=True)
        coreset_list = []
        for i in range(n):
            coreset_path = os.path.join(
                self.save_datasets_dir, f"coreset_dcp_{dpc}_version_{i}.json"
            )
            if os.path.exists(coreset_path):
                logger.info(f"Load coreset[{i}] from {coreset_path}")
                coreset_dataset = Dataset.from_json(coreset_path)
            else:
                logger.info(f"Generate coreset[{i}]")
                coreset_dataset = self.get_coreset(self.dataset, dpc=dpc, seed=i)
                coreset_dataset.to_json(coreset_path)
            coreset_list.append(coreset_dataset)

        return coreset_list

    def get_coreset(self, dataset: Dataset, dpc: int, seed: int = 42) -> Dataset:
        label_datasets = [
            dataset.filter(lambda ex: ex["labels"] == i) for i in range(self.num_labels)
        ]
        label_coresets = [
            self.select(label_dataset, dpc=dpc, seed=seed)
            for label_dataset in label_datasets
        ]
        assert len(label_coresets) == self.num_labels
        return concatenate_datasets(label_coresets)

    def select(self, dataset: Dataset, dpc: int, seed: int) -> Dataset:
        if len(dataset) < dpc:
            warnings.warn(
                f"dataset size ({len(dataset)}) < dpc ({dpc}), use all examples"
            )
            return dataset

        if self.config.coreset_type == "random":
            return random_selection(dataset, dpc=dpc, seed=seed)
        elif self.config.coreset_type == "k_centers":
            return k_centers(
                dataset,
                dpc=dpc,
                model=self.encoder,
                tokenizer=self.encoder_tokenizer,
                sentence_keys=DATASET_ATTRS[self.task_name]["sentence_keys"],
                seed=seed,
            )
        elif self.config.coreset_type == "herding":
            return herding(
                dataset,
                dpc=dpc,
                model=self.encoder,
                tokenizer=self.encoder_tokenizer,
                sentence_keys=DATASET_ATTRS[self.task_name]["sentence_keys"],
            )
        elif self.config.coreset_type == "rank_dilm":
            assert self.generator is not None
            return rank_with_dilm(
                dataset,
                dpc=dpc,
                generator=self.generator,
                sentence_keys=DATASET_ATTRS[self.task_name]["sentence_keys"],
            )
        else:
            raise NotImplementedError
