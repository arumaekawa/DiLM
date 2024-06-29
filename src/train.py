import glob
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from coreset import CoresetConfig, CoresetModule
from data import DataConfig, DataModule
from distillation import DistilledDataConfig, TrainConfig, get_trainer
from evaluator import EvaluateConfig, Evaluator
from generator import GeneratorConfig, GeneratorModel
from learner import LearnerConfig, LearnerModel
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str  # "dilm" or "coreset"
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
    train: TrainConfig
    distilled_data: DistilledDataConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        if os.path.exists(config.train.save_model_dir):
            raise ValueError(
                f"Output directory `{config.train.save_model_dir}` already exists."
            )
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
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


@hydra.main(config_path="../configs/train", config_name="dc", version_base=None)
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
    logger.info(f"Building Learner model: (`{config.learner.model_name}`)")
    learner = LearnerModel(config.learner, task_name=config.data.task_name)

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data, generator=generator, learner=learner)

    # Evaluator
    evaluator = Evaluator(config.evaluate, task_name=config.data.task_name)

    assert config.base.method == "dilm"

    # Build coreset module
    coreset_module = CoresetModule(
        config.coreset,
        config.data.task_name,
        dataset=data_module.datasets["train"],
        generator=generator if config.coreset.coreset_type == "rank_dilm" else None,
    )

    # Train generator
    trainer = get_trainer(
        config=config.train, distilled_data_config=config.distilled_data
    )
    if not config.train.repset_teacher:
        repset_teachers = None
    else:
        assert (
            config.train.gm_real_dpc
            * config.train.gm_real_grad_accum_step
            % config.train.repset_dpc
            == 0
        )
        repset_teachers = coreset_module.generate_dataset(
            dpc=config.train.repset_dpc, n=config.train.n_repset
        )
        repset_teachers = [
            data_module.preprocess_dataset(repset_teacher)
            for repset_teacher in repset_teachers
        ]

    trainer.fit(
        generator=generator,
        learner=learner,
        data_module=data_module,
        evaluator=evaluator,
        repset_teachers=repset_teachers,
        coreset_module=coreset_module,
    )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
