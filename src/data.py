import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, disable_progress_bar, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

from dataset_attrs import DATASET_ATTRS
from generator import GeneratorModel
from learner import LearnerModel

logger = logging.getLogger(__name__)

disable_progress_bar()


@dataclass
class DataConfig:
    task_name: str
    datasets_path: Path
    preprocessed_datasets_path: Path
    train_batch_size: int = 64
    valid_batch_size: int = 256
    test_batch_size: int = 256
    num_proc: int = 1
    force_preprocess: bool = False


class DataModule:
    """DataModule class
    ```
    data_module = DataModule(config.data)
    # preprocess training dataset
    data_module.run_preprocess(generator=generator, learner=learner)
    # preprocess external dataset (for generated data)
    data_module.preprocess_dataset(
        dataset=self.datasets, generator=generator, learner=learner
    )
    ```
    """

    def __init__(
        self, config: DataConfig, generator: GeneratorModel, learner: LearnerModel
    ):
        self.config = config
        self.generator = generator
        self.learner = learner

        # load raw dataset
        self.dataset_attr = DATASET_ATTRS[self.config.task_name]
        self.datasets: DatasetDict = self.get_dataset()
        logger.info(f"Datasets: {self.datasets}")

        self.num_labels: int = self.dataset_attr["num_labels"]
        assert self.num_labels == self.datasets["train"].features["labels"].num_classes

        # preprocess datasets
        self.run_preprocess()
        logger.info(f"Preprocessed datasets: {self.preprocessed_datasets}")

        # data collator
        self.data_collator = self.get_data_collator()

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        else:
            assert self.config.task_name in DATASET_ATTRS
            datasets = load_dataset(*self.dataset_attr["load_args"])

            if "validation" not in datasets:
                datasets["validation"] = datasets.pop(
                    self.dataset_attr["test_split_key"]
                )
            assert datasets.keys() >= {"train", "validation"}

            os.makedirs(os.path.dirname(self.config.datasets_path), exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        if self.dataset_attr["problem_type"] == "single_label_classification":
            # rename label_key
            assert self.dataset_attr["label_key"] in datasets["train"].features
            datasets = datasets.rename_column(self.dataset_attr["label_key"], "labels")
        else:
            raise NotImplementedError

        return datasets

    def run_preprocess(self):
        """datasets preprocessing"""

        if (
            os.path.exists(self.config.preprocessed_datasets_path)
            and not self.config.force_preprocess
        ):
            logger.info(
                "Load preprocessed datasets from `{}`".format(
                    self.config.preprocessed_datasets_path
                )
            )
            self.preprocessed_datasets = load_from_disk(
                self.config.preprocessed_datasets_path
            )
            return

        logger.info("Preprocess dataset")
        self.preprocessed_datasets = self.preprocess_dataset(self.datasets)

        logger.info(
            f"Save preprocessed datasets to `{self.config.preprocessed_datasets_path}`"
        )
        os.makedirs(
            os.path.dirname(self.config.preprocessed_datasets_path), exist_ok=True
        )
        self.preprocessed_datasets.save_to_disk(self.config.preprocessed_datasets_path)

    def preprocess_dataset(
        self, dataset: Optional[Dataset | DatasetDict]
    ) -> Dataset | DatasetDict:
        logger.info("Preprocess dataset")
        # sentence keys for task
        sentence_keys = self.dataset_attr["sentence_keys"]

        # get tokenize function
        def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
            # bos_tokens
            if -1 in batch["labels"]:
                bos_tokens = [self.generator.tokenizer.bos_token] * len(batch["labels"])
            else:
                bos_tokens = [self.generator.bos_tokens_map[i] for i in batch["labels"]]

            # sentences
            batch_sentences = [[s.strip() for s in batch[key]] for key in sentence_keys]
            concat_sentences = [
                f" {self.generator.sep_token} ".join(sents)
                for sents in zip(*batch_sentences)
            ]
            batch_sentences_generator = [
                f"{bos_token} {sent} {self.generator.tokenizer.eos_token}"
                for bos_token, sent in zip(bos_tokens, concat_sentences)
            ]

            # tokenize
            batch_generator = self.generator.tokenizer(
                batch_sentences_generator,
                max_length=self.generator.tokenizer.model_max_length,
                truncation=True,
            )

            batch_learner = self.learner.tokenizer(
                *batch_sentences,
                max_length=self.learner.tokenizer.model_max_length,
                truncation=True,
            )
            batch_learner["labels"] = batch["labels"]

            # rename keys
            batch_generator = {f"generator.{k}": v for k, v in batch_generator.items()}
            batch_learner = {f"learner.{k}": v for k, v in batch_learner.items()}

            return batch_generator | batch_learner

        # tokenize
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Tokenize datasets",
        )

        def format_keys(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
            """set meta keys of data"""
            return {
                meta_key: {
                    key.removeprefix(f"{meta_key}."): value
                    for key, value in batch.items()
                    if key.startswith(f"{meta_key}.")
                }
                for meta_key in ("generator", "learner")
            }

        # format meta key for generator and learner
        dataset = dataset.map(
            format_keys,
            batched=False,
            num_proc=self.config.num_proc,
            desc="Set meta_keys of datasets",
        )

        # remove unused columns from datasets
        if isinstance(dataset, Dataset):
            column_names = dataset.column_names
        else:
            column_names = dataset["train"].column_names
        removed_keys = [
            col for col in column_names if col not in ["generator", "learner"]
        ]
        dataset = dataset.remove_columns(removed_keys)

        return dataset

    def get_data_collator(self):
        data_collator_generator = DataCollatorForLanguageModeling(
            tokenizer=self.generator.tokenizer, mlm=False, pad_to_multiple_of=8
        )
        data_collator_learner = DataCollatorWithPadding(
            tokenizer=self.learner.tokenizer, padding="longest", pad_to_multiple_of=8
        )

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {
                "generator": data_collator_generator([ex["generator"] for ex in batch]),
                "learner": data_collator_learner([ex["learner"] for ex in batch]),
            }

        return collate_fn

    def get_train_loader(
        self,
        dataset: Dataset | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        label: int | None = None,
        **kwargs,
    ):
        if dataset is None:
            dataset = self.preprocessed_datasets["train"]

        if label is not None:
            dataset = dataset.filter(
                lambda example: example["learner"]["labels"] == label
            )

        if batch_size is None:
            batch_size = self.config.train_batch_size

        return DataLoader(
            dataset,
            collate_fn=self.data_collator,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )

    def valid_loader(self) -> DataLoader:
        assert "validation" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["validation"],
            collate_fn=self.data_collator,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            drop_last=False,
        )
