# Future-Shot: Genetic Engineering Attribution

## Introduction

This repository contains the code to deploy the model trained for few-shot learning on ["Improving lab-of-origin prediction of genetically engineered plasmids via deep metric learning"](https://www.nature.com/articles/s43588-022-00234-z), adapted to use the [Future-Shot library](https://github.com/fernandocamargoai/future-shot).

## Installation

To install the dependencies, run:

```bash
poetry install
```

## Usage

To pack the BentoML model, run:

```bash
poetry run python -m gea.deployment.pack
```

To build the Bento, run:

```bash
poetry run bentoml build -f gea/deployment/bentofile.yaml
```

To serve the model, run:

```bash
poetry run bentoml serve genetic_engineering_attribution_service:latest
```

To build the Docker image, run:

```bash
poetry run bentoml containerize genetic_engineering_attribution_service:latest
```

To see the API documentation, go to `http://localhost:3000/docs`.

New labs can be added by invoking the `/new_lab` endpoint, passing some plasmids from the new lab and its ID. `artifacts/few_shot_dataset.csv` contains plasmids from the labs that were not used in the training set. One can then invoke `/predict` to predict the lab of origin of a plasmid.

