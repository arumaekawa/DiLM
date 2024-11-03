# DiLM: Distilling Dataset into Language Model

Implementaiton of "DiLM: Distilling Dataset into Language Model for Text-level Dataset Distillation" (accepted by NAACL2024 Findings)".

**Abstract**: Dataset distillation aims to compress a training dataset by creating a small number of informative synthetic samples such that neural networks trained on them perform as well as those trained on the original training dataset. Current text dataset distillation methods create each synthetic sample as a sequence of word embeddings instead of a text to apply gradient-based optimization; however, such embedding-level distilled datasets cannot be used for training other models whose word embedding weights are different from the model used for distillation. To address this issue, we propose a novel text dataset distillation approach, called Distilling dataset into Language Model (DiLM), which trains a language model to generate informative synthetic training samples as text data, instead of directly optimizing synthetic samples. We evaluated DiLM on various text classification datasets and showed that distilled synthetic datasets from DiLM outperform those from current coreset selection methods. DiLM achieved remarkable generalization performance in training different types of models and in-context learning of large language models. Our code will be available at https://github.com/arumaekawa/DiLM.

**Paper**: [arXiv], [NAACL2024 Findings]

## Contents

This repository utilizes [PyTorch](https://pytorch.org/) and modern experiment manager tools, [Hydra](https://hydra.cc/) and [MLflow](https://www.mlflow.org/).

Datasets and pre-trained models are downloaded and used with [Hugging Face](https://huggingface.co/).

### Directory structure

```
.
├── configs
│  ├── test
│  │  ├── coreset.yaml
│  │  ├── dc.yaml
│  │  └── lm.yaml
│  └── train
│     ├── generator
│     │  ├── pretrained_mnli.yaml
│     │  ├── pretrained_qqp.yaml
│     │  └── pretrained_sst2.yaml
│     ├── dc.yaml
│     └── lm.yaml
├── src
│  ├── coreset
│  │  ├── __init__.py
│  │  ├── coreset_base.py
│  │  ├── coreset_utils.py
│  │  ├── herding.py
│  │  ├── k_centers.py
│  │  ├── random.py
│  │  └── rank_dilm.py
│  ├── distillation
│  │  ├── __init__.py
│  │  ├── distilled_data.py
│  │  ├── trainer_base.py
│  │  ├── trainer_dc.py
│  │  └── trainer_lm.py
│  ├── data.py
│  ├── dataset_attrs.py
│  ├── evaluator.py
│  ├── generator.py
│  ├── learner.py
│  ├── test.py
│  ├── train.py
│  └── utils.py
├── README.md
└── requirements.txt
```

## Run Scripts

1. Install packages (Python 3.10)

   ```bash
   $ pip install -r requirements.txt
   ```

2. Run pre-training (LM)

   ```bash
    $ python src/train.py --config-name=lm data.task_name=sst2
   ```

3. Run dataset fine-tuning (Gradient Matching)

   ```bash
    $ python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2
   ```

4. Run evaluation

   ```bash
    $ python src/test.py --config-name=dc data.task_name=sst2 generator.pretrained_model_dir=path/to/pretrained_model_dir
   ```

5. Check the results with MLFlow (http://localhost:5000)

   ```bash
    $ mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
   ```

## Citation

```
@inproceedings{maekawa-etal-2023-dataset,
    title = "Dataset Distillation with Attention Labels for Fine-tuning {BERT}",
    author = "Maekawa, Aru  and
      Kobayashi, Naoki  and
      Funakoshi, Kotaro  and
      Okumura, Manabu",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.12",
    doi = "10.18653/v1/2023.acl-short.12",
    pages = "119--127",
}
```
