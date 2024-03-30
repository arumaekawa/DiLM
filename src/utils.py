import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import BatchEncoding, SchedulerType, get_scheduler


def average(
    inputs: list[int | float | list | dict | list], std: bool = False
) -> int | float | list | dict:
    if isinstance(inputs[0], (int, float)):
        if std:
            return (np.mean(inputs), np.std(inputs))
        else:
            return np.mean(inputs)
    elif isinstance(inputs[0], list):
        return [average([*ls], std=std) for ls in zip(*inputs)]
    elif isinstance(inputs[0], dict):
        return {k: average([dc[k] for dc in inputs], std=std) for k in inputs[0].keys()}
    else:
        raise TypeError


def log_params_from_omegaconf_dict(params):
    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f"{parent_name}.{k}", v)
                else:
                    mlflow.log_param(f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f"{parent_name}.{i}", v)

    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def batch_to_cuda(batch: dict[str, torch.Tensor] | BatchEncoding):
    return {k: v.cuda() for k, v in batch.items()}


def endless_dataloader(data_loader, max_iteration=1000000):
    for _ in range(max_iteration):
        for batch in data_loader:
            yield batch

    assert False, "Reach max iteration"


def configure_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_type: str,
    scheduler_type: str | SchedulerType,
    weight_decay: float,
    warmup_ratio: float,
    num_train_steps: int,
) -> tuple[Optimizer, LRScheduler]:

    optimizer_class = {"sgd": SGD, "momentum": SGD, "adam": Adam, "adamw": AdamW}
    assert optimizer_type in optimizer_class

    if optimizer_type == "adamw":
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        grouped_params = model.parameters()

    if optimizer_type == "momentum":
        optimizer = optimizer_class[optimizer_type](grouped_params, lr=lr, momentum=0.9)
    else:
        optimizer = optimizer_class[optimizer_type](grouped_params, lr=lr)

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_train_steps * warmup_ratio,
        num_training_steps=num_train_steps,
    )
    return optimizer, scheduler
