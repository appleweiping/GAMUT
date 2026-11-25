# GAMUT: project brief

## Problem

FLARE forecasts the (g/r) magnitudes of ZTF transients from the first 5--30 days with gradient-boosted point and quantile models, and issues conformal 68% and 95% bands. The bands are close to nominal over the population but under-cover rare classes at some forecast horizons because the conformal multiplier is selected using the predicted class. GAMUT targets class-conditional forecast coverage when the class is still uncertain, while retaining competitive band width and the public FLARE folds.

## Method

GAMUT is one calibration method with two fitting protocols. The default cross-fitted protocol reuses the calibration objects and is evaluated empirically. A strict train/calibration/test split fits nuisance quantities and thresholds on disjoint data; this protocol supplies the finite-sample class--horizon guarantee reported in the paper.

## Scope and evaluation

The evaluation uses public data only: the ZTF-BTS benchmark, PLAsTiCC releases, UCR archives, and a synthetic family. It reports per-class coverage, horizon-wise coverage, band width, ablations, calibration diagnostics, and held-out checks against published baselines. The shipped FLARE classifier and forecaster are unchanged.

## Reproducibility constraints

The experiments are designed for CPU execution with public inputs. Complete preparation, evaluation, table-generation, and reproduction instructions are in `README.md`, `scripts/`, `results/`, and `paper/`.

## Design scope

The project does not introduce another deep generative forecaster or change FLARE's service decisions. Claims are restricted to the evaluated datasets, horizons, confidence levels, and fitting protocols.
