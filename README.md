# My Personal MLOps Project

This is a personalized version of an MLOps pipeline, modified to train a **Breast Cancer Classifier** algorithm rather than the typical Iris dataset.

## Setup
You can use standard pip or conda to configure the environment:
```bash
conda env create -f environment.yml
conda activate env_rl_project
```

## Workflows incorporated:
- **ML Model CI:** Runs Linter, Dry Tests, & Uploads documents as artifacts on PR and feature branch pushes.
- **ML End-to-End Pipeline:** Runs `train.py`, evaluates the accuracy threshold via `check_threshold.py`, and triggers a "Mock build" containerization via `Dockerfile` only if metrics pass.
