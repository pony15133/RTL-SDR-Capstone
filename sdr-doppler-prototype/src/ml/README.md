# Machine Learning

This folder contains the Random Forest training and evaluation workflow.

## Purpose

- `train.py` validates a labelled CSV, trains the model, and saves it with metadata.
- `evaluation.py` computes precision, recall, F1, accuracy, confusion matrix, and related metrics.
- `model.py` stores and loads the trained model bundle and metadata.

## Important Note

The current implementation is suitable for pipeline validation and experimentation, but models trained only on synthetic data are not considered scientifically validated real-world models.
