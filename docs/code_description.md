# Code Description

This document summarizes the purpose of each scripted component and the
impact of configurable parameters across the repository.

## `scripts/prep_readgssi.py`
* Converts `.dzt` radargrams into normalized arrays and PNG snapshots.
* Manifest creation captures the relationship between source and output
  files.
* The `--seed` option shifts the deterministic randomization applied to
  placeholder arrays, enabling reproducible variance during tests.

## `scripts/synth_gprmax.py`
* Generates synthetic gprMax-like grids using seeded Gaussian noise.
* Supports reproducible scene creation through `--seed` and coverage
  control via `--count` and `--size`.
* Produces a manifest with per-scene metadata for downstream audits.

## `scripts/train.py`
* Simulates a training loop by mapping hyperparameters to deterministic
  accuracy and loss values.
* Captures results inside `metrics.json` to mimic Lightning and MLflow
  integrations without long-running training.
* Parameter changes (`--epochs`, `--lr`, `--batch`) directly influence
  the produced metrics to illustrate expected sensitivity trends.

## `scripts/eval.py`
* Loads lightweight prediction and label summaries to compute stubbed
  precision, recall, and F1 metrics.
* Designed to validate reporting pipelines while guarding against
  divide-by-zero issues through safe denominators.
* Stores the resulting `evaluation.json` for reproducibility and review.

## `scripts/infer.py`
* Reads arbitrary binary or text inputs to estimate a confidence score
  derived from byte magnitudes.
* Demonstrates how inference outputs may be serialized as JSON for
  integration with monitoring or downstream analytics.
* Provides deterministic behaviour by relying solely on input content,
  making it suitable for smoke tests.

## Dependencies
* `numpy`: required for array manipulations within preprocessing and
  synthetic scene generation.

## Coding Standards
* Every code line contains comments describing how parameter changes
  influence outputs to support handoffs between contributors.
* Line lengths are limited to 73 characters, ensuring compliance with
  the specified formatting constraint.
