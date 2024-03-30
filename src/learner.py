import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from dataset_attrs import DATASET_ATTRS

logger = logging.getLogger(__name__)

AUTO_MODEL_CLASSES = {"single_label_classification": AutoModelForSequenceClassification}

MODEL_ATTRS = {
    "bert-base-uncased": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "classifier_module_names": ["classifier"],
    },
    "bert-large-uncased": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "classifier_module_names": ["classifier"],
    },
    "roberta-base": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "classifier_module_names": ["classifier"],
    },
    "roberta-large": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "classifier_module_names": ["classifier"],
    },
    "xlnet-base-cased": {
        "dropout_keys": [
            "dropout",
            "summary_last_dropout",
        ],
        "classifier_module_names": ["sequence_summary", "logits_proj"],
    },
    "gpt2": {
        "dropout_keys": [
            "embd_pdrop",
            "attn_pdrop",
            "resid_pdrop",
            "summary_first_dropout",
        ],
        "classifier_module_names": ["score"],
    },
}


@dataclass
class LearnerConfig:
    """Config for Learner Model"""

    model_name: str = "bert-base-uncased"
    use_pretrained_model: bool = True
    disable_dropout: bool = False
    gradient_checkpointing: bool = False
    freeze_bert: bool = False

    few_shot: bool = False


class LearnerModel(nn.Module):
    def __init__(self, config: LearnerConfig, task_name: str):
        super().__init__()
        self.config = config
        self.problem_type = DATASET_ATTRS[task_name]["problem_type"]
        self.num_labels = DATASET_ATTRS[task_name]["num_labels"]

        assert self.problem_type != "single_label_classification" or self.num_labels > 1

        if self.config.disable_dropout:
            dropout_configs = {
                dropout_key: 0.0
                for dropout_key in MODEL_ATTRS[self.config.model_name]["dropout_keys"]
            }
        else:
            dropout_configs = {}

        self.bert_model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,
            finetuning_task=task_name,
            problem_type=self.problem_type,
            **dropout_configs,
        )
        model_class = AUTO_MODEL_CLASSES[self.problem_type]
        self.bert_model: PreTrainedModel = model_class.from_pretrained(
            config.model_name,
            from_tf=bool(".ckpt" in config.model_name),
            config=self.bert_model_config,
        )

        if self.config.use_pretrained_model:
            self.initial_state_dict = self.bert_model.state_dict()
            self.classifier_module_names = MODEL_ATTRS[self.config.model_name][
                "classifier_module_names"
            ]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, model_max_length=512
        )
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.bert_model_config.pad_token_id = self.tokenizer.pad_token_id

        if self.config.gradient_checkpointing:
            if self.config.model_name in ["xlnet-base-cased"]:
                logger.warning(
                    f"{self.config.model_name} does not support gradient checkpointing."
                    " Disable gradient checkpointing."
                )
            else:
                self.bert_model.gradient_checkpointing_enable()

        if self.config.freeze_bert:
            for _, p in self.named_parameters_for_bert():
                p.requires_grad = False

    def forward(self, *args, **kwargs) -> SequenceClassifierOutput:
        labels: torch.LongTensor = kwargs.pop("labels") if "labels" in kwargs else None

        outputs: SequenceClassifierOutput = self.bert_model(*args, **kwargs)

        loss = None
        if labels is not None:
            if self.problem_type != "single_label_classification":
                raise NotImplementedError

            if outputs.logits.shape == labels.shape:
                # labels: (batch_size, num_labels) or (batch_size)
                labels = labels.view(-1, self.num_labels)
            else:
                assert labels.ndim == 1, f"{labels.shape=}"

            loss = F.cross_entropy(
                outputs.logits.view(-1, self.num_labels), labels, reduction="none"
            )
            assert loss.shape == labels.shape[:1]  # (batch_size,)

        return SequenceClassifierOutput(
            loss=loss,  # (batch_size,)
            logits=outputs.logits,  # (batch_size, num_labels)
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        return self.bert_model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.bert_model.get_input_embeddings()

    def init_weights(self):
        """init_weights
        Initialize additional weights of pretrained model in the same way
        when calling AutoForSequenceClassification.from_pretrained()
        """

        if not self.config.use_pretrained_model:
            assert hasattr(self.bert_model, "init_weights")
            self.bert_model.init_weights()
        else:
            self.bert_model.load_state_dict(self.initial_state_dict)
            for module_name in self.classifier_module_names:
                initialized_module = self.bert_model
                for p in module_name.split("."):
                    initialized_module = getattr(initialized_module, p)
                for module in initialized_module.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(
                            mean=0.0, std=self.bert_model.config.initializer_range
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif len(list(module.parameters(recurse=False))) > 0:
                        raise NotImplementedError

    def classifier_param_names(self):
        for module_name in self.classifier_module_names:
            module: nn.Module = getattr(self.bert_model, module_name)
            name_prefix = f"bert_model.{module_name}"
            for name, _ in module.named_parameters():
                yield f"{name_prefix}.{name}"

    @property
    def device(self):
        return self.bert_model.device


class LearnerModelForFewShot:
    """
    Learner model for few-shot learning.
    """

    def __init__(self, config: LearnerConfig, task_name: str):

        self.config = config
        self.problem_type = DATASET_ATTRS[task_name]["problem_type"]
        self.num_labels = DATASET_ATTRS[task_name]["num_labels"]

        assert self.problem_type != "single_label_classification" or self.num_labels > 1

        # setup model
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        self.model.eval()

        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, model_max_length=1024, padding_side="right"
        )

        # set pad token
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # resize token embeddings
        self.resize_token_embeddings(len(self.tokenizer))

        # freeze model
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def __call__(self, *args, **kwargs) -> SequenceClassifierOutput:
        """
        Forward method for CausalLM
        """

        if "labels" in kwargs:
            kwargs.pop("labels")

        return self.model(*args, **kwargs)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @property
    def device(self):
        return self.model.device


def get_learner(config: LearnerConfig, task_name: str) -> LearnerModel:
    """
    Get Learner model from config.
    """

    logger.info(f"Building Learner model: (`{config.model_name}`)")

    if config.few_shot:
        return LearnerModelForFewShot(config, task_name)

    return LearnerModel(config, task_name)
