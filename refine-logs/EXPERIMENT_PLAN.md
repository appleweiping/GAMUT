# Experiment Plan (ARIS W1.5)

**Problem.** Forecast bands whose coverage holds for every latent class (true class known in calibration, uncertain
at test time), for multi-horizon light-curve and time-series forecasting.
**Method thesis.** The optimal band under one coverage constraint per class is a level set of a dual-weighted posterior
mixture of class-conditional predictive densities. PRISM estimates the three ingredients: hypothesis-conditional
forecasts, calibrated membership, and dual weights. It calibrates the level so that every class reaches 1 − α.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1 validity | The PI's open problem: SLSN/CV under-coverage | Worst-class coverage ≥ nominal − 0.02 (68%) and − 0.015 (95%) on ≥ 6 benchmarks, at every origin | B1, B5 |
| C2 efficiency | Validity is trivial with wide bands | Smallest class-averaged IS and width among valid methods on ≥ 6 benchmarks | B1, B4 |
| C3 components | Coupled design, not a bag of tricks | Removing any of C1/C2/C3 degrades validity or IS on most benchmarks | B2 |
| C4 theory | Explains why X-measurable calibration fails | Synthetic: the observed gap matches Prop. 2; PRISM matches the oracle length | B3 |
| C5 robustness | Deployment: classifier drift, prior shift, small calibration sets | Coverage holds under degraded posteriors (validity does not need a correct posterior) | B6 |

## Blocks

- **B0 Observation (MUST).** FLARE bands on ZTF at every origin: per-class coverage split by classifier correctness and
  by origin. Gate: confirms the PI's statement (SLSN under-coverage, CV at later origins). Done at h = 5 (pilot).
- **B1 Main comparison (MUST).** 6 benchmarks × 5 origins × {68%, 95%} × 14 methods. Leave-one-fold-out over FLARE's folds.
  Metrics: the full list in the research contract. Paired object-level bootstrap (1,000 resamples) for PRISM vs. the
  best valid baseline on class-averaged IS.
- **B2 Ablation (MUST).** Without C1 (pooled forecaster); C2 replaced by uniform or argmax posterior, or the
  uncalibrated posterior; C3 replaced by a posterior mixture (marginal threshold) or by prior-ratio weights
  (no dual). Plus PRISM-cert (split certificate, point and object units).
- **B3 Synthetic (MUST).** Gaussian class mixture with a known posterior; vary the classifier's accuracy. Oracle set
  from Prop. 1; check that PRISM's length tends to the oracle length and FLARE-style coverage follows Prop. 2.
- **B4 Hyperparameters on validation (MUST).** Nested: inner leave-one-fold-out on the 4 calibration folds of each
  outer fold; the test fold is never used. Grid: min_bin ∈ {50, 150, 500, ∞}; bandwidth × {0.5, 1, 2}; temperature
  {off, on}; unit {point, object}; finite-sample correction {off, on}; prune ∈ {1e-4, 1e-3, 1e-2}.
- **B5 Per-horizon curves (MUST).** Coverage and width per lead-time bin, per class, for FLARE, PRISM and the best baseline.
- **B6 Robustness (NICE).** Posterior temperature distortion (×0.5, ×2), label-noise posteriors, calibration subsampling
  (25/50/100%), PLAsTiCC train → DDF prior shift.
- **B7 Runtime (NICE).** Seconds per fold, for each method.
- **B8 Next-observation closure probability (NICE).** Calibrate FLARE's "one more observation closes the set"
  probability (isotonic / Venn-Abers on cross-fitted outputs); report ECE and Brier.

## Run order

| Milestone | Goal | Gate |
|---|---|---|
| M0 | Cross-fits (all benchmarks, origins) | Finite metrics, fold sizes as FLARE |
| M1 | Baselines reproduced on ZTF h5 | FLARE marginal ≈ nominal, SLSN under-covered |
| M2 | PRISM on all benchmarks | C1/C2 criteria |
| M3 | Ablations + hyperparameters | C3 criterion |
| M4 | Synthetic + robustness + runtime | C4, C5 |
| M5 | Paper | Claim audit passes |

## Compute

4 CPU cores, no GPU. Cross-fits take about 20 min per (benchmark, origin) on ZTF. Evaluations take minutes per
(benchmark, origin, level).

## Risks

- PRISM under-covers a rare class with fewer than 50 calibration objects (e.g. ECG5000 class of 24). Mitigation: the
  object-count finite-sample correction; PRISM-cert; report it in the appendix if it remains.
- MDCP or Kandinsky matches PRISM in width. Mitigation: none needed; report it honestly and position PRISM's
  advantages: no posterior-correctness requirement and no test-time QR.

## Round-4 response: pre-registered held-out evaluation (registered 2026-11-23, before any chunk-02 data was read)

Review round 4 noted that several design choices were revised after test results had been seen and asked for an
evaluation on an untouched set. Registered before downloading the data:

- **Data.** PLAsTiCC unblinded test chunk 02 (`plasticc_test_lightcurves_02.csv.gz`, Zenodo 2539456), wide-fast-deep
  objects (`ddf_bool = 0`). No object of this chunk has been used in any earlier run (chunk 01 is DDF-only).
- **Pipeline.** Identical to `plasticc_ddf`: FLARE quality cut, classes with at least 30 objects, five stratified folds,
  origins 5/15/30 d, the same cross-fit recipe (`--lr 0.1 --cap 16`). A cheap detection-count pre-filter equivalent to
  the quality cut is applied before feature extraction, for speed only.
- **Code frozen.** GAMUT v8 / grid g2; md5 `gamut/methods.py` a4e1cb90e66749e5e7ddd37069a9ceb9,
  `gamut/metrics.py` 1febdfedc18aac75c9b1aeefa99e9e13. `gamut/latent_baselines.py` only gains the "+ C1" baseline
  variants (no change to existing baselines). No hyperparameter is tuned on this set.
- **Run once.** All methods and ablations, both levels; the same bootstrap validity rule. The results are reported
  whatever they are: a summary in the main text and full tables in the appendix. No design change follows from them in
  this paper.

## Closure probability (the professor's second open problem), registered 2026-11-23 before any computation

FLARE's `next_observation_value` reports p_singleton, the fraction of 16 simulated next-night magnitudes (Gaussian
around the pooled, predicted-class-routed forecast, fixed scatter per band) for which the classifier's 90% conformal
set becomes a singleton. It ranks objects but is not calibrated.

- **Data / split.** FLARE's own splits: the deployed day-5 and day-10 classifiers were trained on `train`; we fit every
  calibrator on `val` and report on `test` only. Event: the object's real next g or r detection after the origin,
  within 2 days ("next observation"), appended to the prefix; outcome = the deployed 90% set becomes a singleton.
- **Scores.** (S0) deployed p_singleton as shipped (draws at h+1); (S1) the same at the real next time;
  (S2, ours) the latent-class mixture p_close = sum_c p_hat_c(x) P(singleton | next point ~ class-c law), with GAMUT's C1
  hypothesis-conditional forecaster (out-of-fold) and class-c empirical standardized residuals from `val` (8 quantile
  draws per class, classes with p_hat_c >= 0.01), and p_hat the temperature-scaled deployed posterior (C2, fitted on val).
- **Calibration layers** fitted on `val`: none, Platt, isotonic, inductive Venn-Abers (IVAP).
- **Metrics on `test`.** ECE (15 equal-mass bins), Brier, log loss, AUC, calibration slope/intercept, per-true-class
  ECE and Brier, reliability diagrams; object-level bootstrap intervals. Subsets: all objects, and objects whose set
  before has more than one class.
- **Primary claim to test.** S2 + IVAP has lower test ECE and Brier than S0 (and S0 + IVAP), without lower AUC.
  Results are reported whatever they are.

## GAMUT-Q: interval-score variant (registered 2026-11-23, before any implementation)

Motivation (test-informed): on the registered efficiency endpoint, the class-averaged interval score (IS), GAMUT is
significantly worse than the best valid baseline in 13/20 (68%) and 14/20 (95%) main settings. GAMUT minimises band
length; IS also penalises the distance of misses, which density level sets do not control.

- **Variant.** Same C1 (residual laws), C2 (temperature) and C3 (dual fixed point on per-class object-level CRC
  thresholds, adaptive horizon cells). Only the score changes: s_v(x, y) = min(F_v(y|x), 1 - F_v(y|x)), with
  F_v the CDF of the dual-weighted posterior mixture sum_c v_c p_c(x) F_c(y|x) / sum_c v_c p_c(x). The band is the
  central interval [F_v^{-1}(t), F_v^{-1}(1 - t)].
- **Development.** The six main benchmarks are used for development only; anything measured there is test-informed
  and labelled as such.
- **Confirmation (the only basis for a claim).** A fresh held-out set never read before: wide-fast-deep objects of
  the unblinded PLAsTiCC test chunk 03, same pipeline as PLAsTiCC-WFD (chunk 02), origins 5/15/30 d, frozen code, run
  once. Primary claim: GAMUT-Q is valid (bootstrap rule) at every origin and level and has a lower class-averaged IS
  than every valid baseline in the paired bootstrap in a majority of the 6 settings. Reported whatever it is.
- **Code frozen for the chunk-03 confirmation** (recorded before chunk 03 was read): md5 gamut/methods.py a4e1cb90e66749e5e7ddd37069a9ceb9,
  gamut/variants.py (GAMUT-Q) aa2bf36a4b5d40513411669c5a622192, gamut/metrics.py 1febdfedc18aac75c9b1aeefa99e9e13, gamut/latent_baselines.py 7eeb7ba86a74140142a90419e39963eb. No change to GAMUT-Q
  after the development runs on the main benchmarks (they started at the same time as this freeze).
