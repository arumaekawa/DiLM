import logging

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from .coreset_utils import get_embeddings, l2_dist

logger = logging.getLogger(__name__)


@torch.no_grad()
def herding(
    dataset: Dataset,
    dpc: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sentence_keys: list[str],
):
    assert len(dataset) >= dpc

    embeddings = get_embeddings(dataset, model, tokenizer, sentence_keys)
    embeddings = embeddings.cuda()

    indices = torch.arange(embeddings.size(0), device="cuda")
    select_results = torch.zeros(len(dataset), dtype=torch.bool, device="cuda")

    mean_original = embeddings.mean(0)

    logger.info("Selecting samples with herding")
    for i in range(dpc):
        assert sum(select_results) == i
        if i == 0:
            sum_selected = torch.zeros_like(mean_original).unsqueeze(0)
        else:
            sum_selected = embeddings[indices[select_results]].sum(0, keepdim=True)

        dists = l2_dist(
            sum_selected + embeddings[indices[~select_results]],
            mean_original * (i + 1),
        )
        select_results[indices[~select_results][dists.argmin().item()]] = True

    selected_indices = indices[select_results].tolist()
    assert len(selected_indices) == dpc
    return dataset.select(selected_indices)
