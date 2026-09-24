# Features

This folder contains the shared feature-extraction pipeline.

## Purpose

The code here converts a spectrogram into a fixed feature vector that is used by both the rule-based detector and the ML model.

## Key Role

- keeps one source of truth for feature math
- prevents feature drift between detection and training
- ensures training and runtime inference use the same schema

The main output is a `FeatureVector` object with consistent field names (`FEATURE_NAMES`).
