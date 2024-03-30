from .distilled_data import DistilledDataConfig
from .trainer_base import TrainConfig, TrainerBase
from .trainer_dc import TrainerDC
from .trainer_lm import TrainerLM

__all__ = ["TrainerConfig", "get_trainer", "DistilledDataConfig"]

TRAINER_CLASSES = {
    "lm": TrainerLM,
    "dc": TrainerDC,
}


def get_trainer(
    config: TrainConfig, distilled_data_config: DistilledDataConfig
) -> TrainerBase:
    assert config.train_type in TRAINER_CLASSES
    return TRAINER_CLASSES[config.train_type](config, distilled_data_config)
