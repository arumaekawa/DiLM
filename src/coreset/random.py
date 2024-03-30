import random

from datasets import Dataset


def random_selection(dataset: Dataset, dpc: int, seed: int) -> Dataset:
    random.seed(seed)

    assert len(dataset) >= dpc
    selected_sample_ids = random.sample(range(len(dataset)), dpc)

    return dataset.select(selected_sample_ids)
