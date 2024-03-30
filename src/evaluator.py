import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import evaluate
import mlflow
import numpy as np
import torch
from datasets import Dataset
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import DataCollatorWithPadding

from data import DataModule
from dataset_attrs import DATASET_ATTRS
from learner import LearnerModel, LearnerModelForFewShot
from utils import average, batch_to_cuda, configure_optimizer, endless_dataloader

logger = logging.getLogger(__name__)


class Metric:
    """Metric class
    >>> metric = Metric(config.data.task_name)
    >>> metric.add_batch(logits, labels)
    >>> results = metric.compute()
    """

    def __init__(self, task_name: str):
        self.metric = evaluate.load(*DATASET_ATTRS[task_name]["metric_args"])
        self.preprocess = preprocess_for_classification
        self.metric_key = DATASET_ATTRS[task_name]["metric_key"]

    def add_batch(self, logits: torch.Tensor, labels: torch.Tensor):
        return self.metric.add_batch(**self.preprocess(logits, labels))

    def compute(self) -> dict[str, float]:
        results = self.metric.compute()
        if self.metric_key == "combined_score":
            assert len(results) > 1
            results["combined_score"] = np.mean(list(results.values())).item()
        return results


def preprocess_for_classification(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, list[int]]:
    assert logits.ndim == 2
    assert labels.ndim == 1
    return {"predictions": logits.argmax(-1).tolist(), "references": labels.tolist()}


@dataclass
class EvaluateConfig:
    task_name: str
    n_eval_dataset: int = 5
    n_eval_per_dataset: int = 4
    fp16: bool = False
    bf16: bool = False
    save_result_dir: str = "save/result/dir"

    # training config
    few_shot: bool = False

    # finetune config
    optimizer_type: str = "adamw"  # ["sgd", "momentum", "adam", "adamw"]
    scheduler_type: str = "linear"
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.5

    train_step: int = 200
    batch_size: int = 128

    def __post_init__(self):
        assert not (self.fp16 and self.bf16)


class Evaluator:
    def __init__(
        self,
        config: EvaluateConfig,
        task_name: str,
    ):
        self.config = config
        self.metric = Metric(task_name)
        self.metric_key = self.metric.metric_key

    def evaluate(
        self,
        dataset_list: list[Dataset],
        learner: LearnerModel,
        data_module: DataModule,
        save_result_dir: str | os.PathLike,
        n_eval_per_dataset: int | None = None,
        verbose: bool = False,
    ) -> list[dict[str, tuple[float]]]:

        all_results = []
        for i, dataset in enumerate(dataset_list):
            logger.info(f"Evaluate Dataset[{i}] " + "-" * 10)
            save_path = os.path.join(save_result_dir, f"dataset_{i}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            logger.info(f"Save dataset in `{save_path}")
            dataset.to_json(save_path)

            dataset = data_module.preprocess_dataset(dataset)
            results = self.evaluate_dataset(
                dataset,
                learner=learner,
                eval_loader=data_module.valid_loader(),
                data_module=data_module,
                n_eval_per_dataset=n_eval_per_dataset,
                verbose=verbose,
            )
            all_results += results

        if verbose:
            avg_results = average(all_results, std=True)
            avg_results = {k: f"{v[0]}±{v[1]}" for k, v in avg_results.items()}
            logger.info(f"Final Results: {avg_results}")

        logger.info(f"Save results in `{save_path}")
        json.dump(all_results, open(os.path.join(save_result_dir, "results.json"), "w"))

        mlflow.log_artifact(save_result_dir)

        return all_results

    def evaluate_dataset(
        self,
        train_dataset: Dataset,
        learner: LearnerModel,
        eval_loader: DataLoader,
        data_module: DataModule,
        n_eval_per_dataset: int | None = None,
        verbose: bool = False,
    ) -> list[dict[str, float]]:
        if n_eval_per_dataset is None:
            n_eval_per_dataset = self.config.n_eval_per_dataset

        results_for_dataset = []
        for i in trange(
            n_eval_per_dataset, dynamic_ncols=True, leave=False, desc="Evaluate dataset"
        ):
            train_loader = data_module.get_train_loader(
                train_dataset,
                batch_size=min(self.config.batch_size, len(train_dataset)),
                shuffle=True,
                drop_last=True,
            )
            train_loader = endless_dataloader(
                train_loader, max_iteration=self.config.train_step
            )

            learner.init_weights()
            results = self.train_learner(learner=learner, train_loader=train_loader)
            results = self.evaluate_learner(learner, eval_loader)
            if verbose:
                logger.info(f"Model[{i}]: {results}")
            results_for_dataset.append(results)

        return results_for_dataset

    def train_learner(
        self, learner: LearnerModel, train_loader: DataLoader
    ) -> dict[int, dict[str, float]]:

        learner.cuda()

        optimizer, scheduler = configure_optimizer(
            learner,
            lr=self.config.lr,
            optimizer_type=self.config.optimizer_type,
            scheduler_type=self.config.scheduler_type,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            num_train_steps=self.config.train_step,
        )
        learner.train()

        for _ in trange(
            self.config.train_step,
            leave=False,
            dynamic_ncols=True,
            desc="Train learner",
        ):

            batch = next(train_loader)
            # compute loss
            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                loss = learner(**batch_to_cuda(batch["learner"])).loss.mean()

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # gradient clipping
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    learner.parameters(), self.config.max_grad_norm
                )
            # update parameter
            optimizer.step()
            scheduler.step()

    @torch.inference_mode()
    def evaluate_learner(
        self, learner: LearnerModel, data_loader: DataLoader
    ) -> dict[str, float]:
        learner.eval()

        total_loss, num_samples = 0, 0
        for batch in tqdm(
            data_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner"
        ):
            batch = batch["learner"]
            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = learner(**batch_to_cuda(batch))
            assert outputs.loss.shape == (len(batch["labels"]),)

            self.metric.add_batch(outputs.logits, batch["labels"])
            total_loss += outputs.loss.sum().item()
            num_samples += len(batch["labels"])

        results = self.metric.compute()
        results["loss"] = total_loss / num_samples

        return results

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16


def example_to_prompt(
    example: dict[str, Any],
    label_dict: dict[int, str],
    sentence_keys: list[str],
    without_label: bool = False,
) -> str:
    """
    Create prompt for a single example.

    Return:
        if without_label is True:
            sentence_key1: {sentence1}
            sentence_key2: {sentence2}
            label:
        elif without_label is False:
            sentence_key1: {sentence1}
            sentence_key2: {sentence2}
            label: {label}
    """
    example_prompt = ""
    for sentence_key in sentence_keys:
        assert sentence_key in example.keys()
        example_prompt += f"{sentence_key}: {example[sentence_key].strip()}\n"

    example_prompt += "label:"

    if not without_label:
        assert "labels" in example.keys()
        example_prompt += " " + label_dict[example["labels"]] + "\n"

    return example_prompt


def create_train_prompt(
    dataset: Dataset,
    sentence_keys: list[str],
    label_dict: dict[int, str],
):
    """
    Create prompt with few-shot training examples.
    """

    # create train prompt
    train_prompt = ""
    for train_example in dataset.shuffle():
        train_prompt += example_to_prompt(train_example, label_dict, sentence_keys)
        train_prompt += "\n"

    return train_prompt


def get_eval_dataset(
    train_dataset: Dataset,
    data_module: DataModule,
    learner: LearnerModelForFewShot,
    label_dict: dict[int, str],
    sentence_keys: list[str],
):
    """
    Preprocess valid dataset for few-shot learning
    """

    # get valid dataset
    eval_dataset = data_module.datasets["validation"]

    # create train prompt
    train_prompt = create_train_prompt(train_dataset, sentence_keys, label_dict)

    # prompt generation function
    def generate_prompt(example):
        eval_example_prompt = example_to_prompt(
            example, label_dict, sentence_keys, without_label=True
        )
        prompt = train_prompt + eval_example_prompt
        return {"prompt": prompt}

    # add prompt to eval_dataset
    eval_dataset = eval_dataset.map(generate_prompt, batched=False)

    # tokenize
    tokenizer = learner.tokenizer

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"], max_length=tokenizer.model_max_length, truncation=True
        )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_module.config.num_proc,
        desc="Tokenize datasets",
    )

    # remove unused columns from datasets
    removed_keys = [
        col
        for col in eval_dataset.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
    eval_dataset = eval_dataset.remove_columns(removed_keys)

    return eval_dataset


def get_eval_loader(
    eval_dataset: Dataset, learner: LearnerModelForFewShot, batch_size: int
):
    """
    Get eval_loader
    """

    # data_collator
    tokenizer = learner.tokenizer
    data_collator = DataCollatorWithPadding(
        tokenizer, padding="longest", max_length=tokenizer.model_max_length
    )

    return DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False
    )


class EvaluatorForFewShot(Evaluator):
    def evaluate(
        self,
        dataset_list: list[Dataset],
        learner: LearnerModelForFewShot,
        data_module: DataModule,
        save_result_dir: str | os.PathLike,
        n_eval_per_dataset: int | None = None,
        verbose: bool = False,
    ) -> list[dict[str, tuple[float]]]:
        """
        Evaluate datasets with in-context few-shot learning.
        """

        # get sentence_keys and label_dict
        sentence_keys = data_module.dataset_attr["sentence_keys"]
        label_dict = data_module.dataset_attr["label_dict"]

        # redefine label_dict
        logger.info(f"label_dict: {label_dict}")

        all_results = []
        for i, dataset in enumerate(dataset_list):
            logger.info(f"Evaluate Dataset[{i}] " + "-" * 10)
            save_path = os.path.join(save_result_dir, f"dataset_{i}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            logger.info(f"Save dataset in `{save_path}")
            dataset.to_json(save_path)

            results = self.evaluate_dataset(
                dataset,
                learner=learner,
                data_module=data_module,
                sentence_keys=sentence_keys,
                label_dict=label_dict,
                n_eval_per_dataset=n_eval_per_dataset,
                verbose=verbose,
            )
            all_results += results

        if verbose:
            avg_results = average(all_results, std=True)
            avg_results = {k: f"{v[0]}±{v[1]}" for k, v in avg_results.items()}
            logger.info(f"Final Results: {avg_results}")

        logger.info(f"Save results in `{save_path}")
        json.dump(all_results, open(os.path.join(save_result_dir, "results.json"), "w"))

        mlflow.log_artifact(save_result_dir)

        return all_results

    def evaluate_dataset(
        self,
        train_dataset: Dataset,
        learner: LearnerModelForFewShot,
        data_module: DataModule,
        sentence_keys: list[str],
        label_dict: dict[int, str],
        n_eval_per_dataset: int | None = None,
        verbose: bool = False,
    ) -> list[dict[str, float]]:
        if n_eval_per_dataset is None:
            n_eval_per_dataset = self.config.n_eval_per_dataset

        results_for_dataset = []
        for i in trange(
            n_eval_per_dataset, dynamic_ncols=True, leave=False, desc="Evaluate dataset"
        ):
            # preprocess eval_dataset with few-shot prompt
            eval_dataset = get_eval_dataset(
                train_dataset, data_module, learner, label_dict, sentence_keys
            )

            # get eval_loader
            eval_loader = get_eval_loader(eval_dataset, learner, self.config.batch_size)

            # evaluate learner
            results = self.evaluate_learner(learner, eval_loader, label_dict)

            if verbose:
                logger.info(f"Model[{i}]: {results}")

            results_for_dataset.append(results)

        return results_for_dataset

    def evaluate_learner(
        self,
        learner: LearnerModelForFewShot,
        data_loader: DataLoader,
        label_dict: dict[int, str],
    ) -> dict[str, float]:
        """
        Evaluate learner with few-shot prompt.
        """

        for batch in tqdm(
            data_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner"
        ):
            assert "labels" in batch.keys()
            assert "input_ids" in batch.keys()
            assert (
                batch["input_ids"].shape[1] <= learner.tokenizer.model_max_length
            ), f"{batch['input_ids'].shape[1]} > {learner.tokenizer.model_max_length}"

            labels = batch.pop("labels")

            # get model prediction probabilities for labels (logits)
            label_logits = self.get_model_probs(learner, batch, label_dict)

            self.metric.add_batch(label_logits, labels)

        results = self.metric.compute()

        return results

    @torch.inference_mode()
    def get_model_probs(
        self,
        learner: LearnerModelForFewShot,
        batch: dict[str, torch.Tensor],
        label_dict: dict[int, str],
    ) -> torch.Tensor:
        """
        Get model prediction probabilities for labels.
        """
        batch_size = len(batch["input_ids"])

        assert (
            batch["input_ids"].shape[1] <= learner.tokenizer.model_max_length
        ), f"{batch['input_ids'].shape[1]} > {learner.tokenizer.model_max_length}"

        # get model output
        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = learner(**batch_to_cuda(batch))

        # get last token position
        pad_token_id = learner.tokenizer.pad_token_id
        last_token_position = batch["input_ids"].ne(pad_token_id).sum(dim=1) - 1
        assert outputs.logits.shape[0] == len(last_token_position)

        next_token_logits: torch.Tensor = outputs.logits[
            torch.arange(batch_size), last_token_position
        ]

        assert next_token_logits.shape == (batch_size, len(learner.tokenizer))

        # get token_id for each label
        if "llama" not in learner.config.model_name:
            label_dict = {k: " " + v for k, v in label_dict.items()}

        label_token_ids = [
            learner.tokenizer.encode(label_name, add_special_tokens=False)[0]
            for label_name in label_dict.values()
        ]

        # check if label_token_ids are unique
        assert len(label_token_ids) == len(set(label_token_ids)), label_token_ids

        # get model prediction probabilities for labels
        label_logits = next_token_logits[:, label_token_ids]

        # normalize logits
        label_logits = torch.log_softmax(label_logits, dim=-1)

        return label_logits


def get_evaluator(config: EvaluateConfig, task_name: str) -> Evaluator:
    """
    Get evaluator
    """

    if config.few_shot:
        return EvaluatorForFewShot(config, task_name)

    return Evaluator(config, task_name)
