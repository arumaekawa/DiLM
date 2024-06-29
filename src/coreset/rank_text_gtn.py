import logging

import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

from generator import GeneratorModel

from .coreset_utils import batch_to_cuda

logger = logging.getLogger(__name__)


def rank_with_dilm(
    dataset: Dataset, dpc: int, generator: GeneratorModel, sentence_keys: list[str]
):
    generator.cuda()
    generator.eval()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=generator.tokenizer, mlm=False, pad_to_multiple_of=8
    )

    def compute_loss(batch: dict):
        batch_size = len(batch[list(batch.keys())[0]])
        if -1 in batch["labels"]:
            bos_tokens = [generator.tokenizer.bos_token] * len(batch["labels"])
        else:
            bos_tokens = [generator.bos_tokens_map[i] for i in batch["labels"]]

        # sentences
        batch_sentences = [[s.strip() for s in batch[key]] for key in sentence_keys]
        concat_sentences = [
            f" {generator.sep_token} ".join(sents) for sents in zip(*batch_sentences)
        ]
        batch_sentences = [
            f"{bos_token} {sent} {generator.tokenizer.eos_token}"
            for bos_token, sent in zip(bos_tokens, concat_sentences)
        ]
        batch = generator.tokenizer(batch_sentences)
        inputs = data_collator(
            [{k: v[i] for k, v in batch.items()} for i in range(batch_size)]
        )
        with torch.inference_mode():
            losses = generator.compute_loss(**batch_to_cuda(inputs))
        assert losses.size() == (batch_size,)
        return {"loss": losses.tolist()}

    logger.info("Computing losses with generator")
    losses = dataset.map(compute_loss, batched=True, batch_size=256)["loss"]
    logger.info("Done!!")

    topk_indices = torch.topk(torch.tensor(losses), k=dpc, largest=False)[1]
    return dataset.select(topk_indices)
