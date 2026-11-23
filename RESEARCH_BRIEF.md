# Research Brief: forecast bands whose coverage holds for every (uncertain) class

## Problem

FLARE (Sasli et al. 2026, arXiv:2610.30656; github.com/applecider-ml/flare) forecasts the g/r magnitudes of ZTF
transients from the first 5-30 days with gradient-boosted point and quantile models, and issues conformal 68% and 95%
bands. The bands reach nominal coverage over the population but under-cover the SLSNe and, at later horizons, the CVs
(shipped metadata: day-50 68% coverage SLSN 0.41, AGN 0.61, TDE 0.86, ALL 0.70; "SLSN and TDE 0.45-0.65 at nominal 0.68").
The conformal multiplier depends on the PREDICTED class, which is often wrong in the first days for rare objects, so the
band is calibrated for the wrong population. Goal (from the FLARE PI): a band whose coverage holds conditionally on the
uncertain class, at every horizon, with competitive width, evaluated on the public benchmark with the same folds,
reporting per-class coverage and width per horizon against the current bands.

## What was tried by the group (do not repeat)

Class weighting, rare-class augmentation, a diffusion model, a neural process, a temporal convolutional network.
Suggested starting points: conditional conformal (Gibbs, Cherian & Candes 2023), conditioning on classifier probabilities
instead of the predicted label, a mixture of class-wise bands weighted by the probabilities.

## Constraints

4 CPU cores, no GPU. Public data only (BTS benchmark of the FLARE repository; PLAsTiCC unblinded release; UCR archive).
Target venue: ICML 2027 (8-page main body, unlimited appendix, Impact Statement). Executor and reviewers are the same model
family (review_independence: same-family; acceptance provisional).

## Quality bar (user rules)

At least six benchmarks and at least seven external baselines (other papers' frameworks) besides ours; a quantitative
observation study motivating the method; ablations showing every component matters; hyperparameter analysis on validation
data; at least three tightly coupled components with fewer than three loss terms; Method about two pages; complete metrics
at several horizons and levels; main text leads with the strengths; every unfavourable result is reported in the appendix.

## Non-goals

No change to FLARE's classifier; no new deep generative forecaster (already tried); no claims beyond the evaluated scope.
