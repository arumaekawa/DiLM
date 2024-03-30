"""
Dataset attributes for loading and processing datasets.
"""


DATASET_ATTRS = {
    "sst2": {
        "load_args": ("glue", "sst2"),
        "sentence_keys": ("sentence",),
        "label_key": "label",
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "num_labels": 2,
        "metric_args": ("glue", "sst2"),
        "max_length": 68,
        "metric_key": "accuracy",
        "label_dict": {0: "negative", 1: "positive"},
    },
    "mnli": {
        "load_args": ("glue", "mnli"),
        "sentence_keys": ("premise", "hypothesis"),
        "label_key": "label",
        "problem_type": "single_label_classification",
        "test_split_key": "validation_matched",
        "num_labels": 3,
        "metric_args": ("glue", "mnli"),
        "max_length": 421,
        "metric_key": "accuracy",
        "label_dict": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
    "qqp": {
        "load_args": ("glue", "qqp"),
        "sentence_keys": ("question1", "question2"),
        "label_key": "label",
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "num_labels": 2,
        "metric_args": ("glue", "qqp"),
        "max_length": 313,
        "metric_key": "combined_score",
        "label_dict": {0: "unequal", 1: "equal"},
    },
}
