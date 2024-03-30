from dataclasses import dataclass


@dataclass
class DistilledDataConfig:
    dpc: int
    n_dataset: int
    over_sample_ratio: float = 1.0  # if > 1.0, prune samples with k_center
    save_dataset_path: str = "path/to/save_dataset_dir"
