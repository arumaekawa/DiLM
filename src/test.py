import glob
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
import mlflow
import torch
from datasets import Dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from coreset import CoresetConfig, CoresetModule
from data import DataConfig, DataModule
from distillation import DistilledDataConfig
from evaluator import EvaluateConfig, get_evaluator
from generator import GeneratorConfig, GeneratorModel
from learner import LearnerConfig, get_learner
from utils import average, log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str  # "dilm" or "coreset"
    run_name: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    coreset: CoresetConfig
    generator: GeneratorConfig
    learner: LearnerConfig
    distilled_data: DistilledDataConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            if os.path.exists(config.evaluate.save_result_dir):
                raise ValueError(
                    "Output directory `{}` already exists.".format(
                        config.evaluate.save_result_dir
                    )
                )
            # device info
            mlflow.log_params(
                {"hostname": os.uname()[1], "device": torch.cuda.get_device_name()}
            )
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add log file
            log_file_name = f"{os.path.basename(__file__).split('.', 1)[0]}.log"
            mlflow.log_artifact(os.path.join(output_dir, log_file_name))
        return out

    return wrapper


@hydra.main(config_path="../configs/test", config_name="dc", version_base=None)
@mlflow_start_run_with_hydra
def main(config: Config):

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # Set seed
    set_seed(config.base.seed)

    # Generator
    logger.info(f"Building Generator model: (`{config.generator.model_name}`)")
    generator = GeneratorModel(config.generator, task_name=config.data.task_name)

    # Learner
    learner = get_learner(config.learner, task_name=config.data.task_name)

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data, generator=generator, learner=learner)

    # Evaluator
    evaluator = get_evaluator(config.evaluate, task_name=config.data.task_name)

    if os.path.exists(config.distilled_data.save_dataset_path):
        # Load generated dataset from saved path
        dataset_list: list[Dataset] = []
        for i in range(config.distilled_data.n_dataset):
            save_path = os.path.join(
                config.distilled_data.save_dataset_path, f"dataset_{i}.json"
            )
            assert os.path.exists(save_path), f"File not found: {save_path}"
            logger.info(f"Load dataset[{i}] in `{save_path}")
            dataset_list.append(Dataset.from_json(save_path))

    else:
        if config.base.method == "dilm":
            # DiLM
            assert config.generator.pretrained_model_dir is not None

            # Build coreset module
            coreset_module = CoresetModule(
                config.coreset,
                config.data.task_name,
                dataset=data_module.datasets["train"],
                generator=generator
                if config.coreset.coreset_type == "rank_dilm"
                else None,
            )

            # Generate distilled dataset with generator
            generate_dpc = int(
                config.distilled_data.dpc * config.distilled_data.over_sample_ratio
            )
            dataset_list = generator.generate_dataset(
                dpc=generate_dpc, n=config.distilled_data.n_dataset
            )
            if config.distilled_data.over_sample_ratio > 1.0:
                # Select from generated example with coreset module
                dataset_list = [
                    coreset_module.get_coreset(dataset, dpc=config.distilled_data.dpc)
                    for dataset in dataset_list
                ]

        elif config.base.method == "coreset":
            # Coreset

            # Select from real example with coreset module
            coreset_module = CoresetModule(
                config.coreset,
                config.data.task_name,
                dataset=data_module.datasets["train"],
                generator=generator
                if config.coreset.coreset_type == "rank_dilm"
                else None,
            )
            dataset_list = coreset_module.generate_dataset(
                dpc=config.distilled_data.dpc, n=config.distilled_data.n_dataset
            )
        else:
            raise NotImplementedError

        # Save generated dataset
        os.makedirs(config.distilled_data.save_dataset_path, exist_ok=True)
        for i, dataset in enumerate(dataset_list):
            save_path = os.path.join(
                config.distilled_data.save_dataset_path, f"dataset_{i}.json"
            )
            logger.info(f"Save dataset in `{save_path}")
            dataset.to_json(save_path)

    logger.info(f"All dataset saved in `{config.distilled_data.save_dataset_path}`")
    mlflow.log_artifact(config.distilled_data.save_dataset_path)

    # Evaluate generated dataset
    results = evaluator.evaluate(
        dataset_list=dataset_list,
        learner=learner,
        data_module=data_module,
        save_result_dir=config.evaluate.save_result_dir,
        verbose=True,
    )

    avg_results = average(results, std=True)
    mlflow.log_metrics({f"avg.{k}": v[0] for k, v in avg_results.items()})
    mlflow.log_metrics({f"std.{k}": v[1] for k, v in avg_results.items()})


if __name__ == "__main__":
    main()
