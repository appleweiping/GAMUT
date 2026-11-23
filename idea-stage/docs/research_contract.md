# Research Contract: PRISM — forecast bands valid for every latent class

review_independence: same-family · acceptance_status: provisional · created 2026-11-22 · venue: ICML 2027

## Selected idea

- **Description.** The light-curve forecast bands of FLARE are calibrated per *predicted* class. Early in a transient's
  life the classifier is often wrong for rare classes, so the band is calibrated for the wrong population and SLSNe
  (and CVs at later origins) are under-covered. PRISM treats the class as a latent variable that is known on the
  calibration set and uncertain at test time. It builds the length-optimal band subject to one coverage constraint
  per true class. That band is a level set of a dual-weighted posterior mixture of class-conditional predictive
  densities, calibrated so that every class reaches 1 − α.
- **Source.** The FLARE PI's request (conditional conformal, conditioning on classifier probabilities, mixture of
  class-wise bands). Pilot recorded in `idea-stage/PILOT_CRITERIA.md`; pilot criterion 2 met at ZTF h = 5 d.
- **Selection rationale.** Every probability-conditioned method leaves SLSN at about 0.54 coverage (pilot):
  GCC on p(x), localized CP in p-space, label-shift mixture and k-means partitions. Union and max methods
  (two-step CP, max-class CP, MDCP) are valid but about 50% wider. PRISM is the only design that uses membership
  and still enforces per-class coverage.

## Answers to the planning questions (decided before the full runs)

| Question | Decision | Why |
|---|---|---|
| Benchmarks | **6 main + 1 appendix + 1 synthetic.** Main: ZTF-BTS (FLARE's public benchmark, same folds), PLAsTiCC-train (11 classes, rare SLSN, TDE, uLens, SNIa-91bg), PLAsTiCC-DDF (deep-drilling test release, a different cadence), UCR StarLightCurves, UCR ECG5000 (classes of 24 and 96 series), UCR ElectricDevices (7 classes, 16.6k series). Appendix: UCR Wafer (2 classes). Synthetic: a known-posterior mixture in which the oracle set is computable. | Three astronomical datasets with real class imbalance and the professor's own benchmark. Three non-astronomical latent-class forecasting tasks that show generality. Each has ≥ 2,000 series and ≥ 2 rare classes, except Wafer. |
| External baselines | **12 frameworks from other papers.** FLARE bands (Sasli et al. 2026); split CP with normalized residuals (Lei et al., JASA 2018); CQR (Romano et al., NeurIPS 2019); DCP (Chernozhukov et al., PNAS 2021); localized CP (Guan, Biometrika 2023); label-shift mixture CP (Podkopaev & Ramdas, UAI 2021); conditional-guarantee CP / GCC (Gibbs, Cherian & Candès, JRSSB 2025); learned-partition CP (Kiyani et al., ICML 2024); two-step CP (Timans et al., ECCV 2024); Kandinsky CP (Bairaktari et al., ICML 2025); CPTC (Sun & Yu, NeurIPS 2025); MDCP (Yang & Jin, ICML 2026). References (not counted): max-class CP (Barber et al. 2021) and the true-class oracle. | Covers the three families: membership-blind (split, CQR, DCP), probability-conditioned (GCC, localized, mixture, partition, CPTC), and latent-class-valid (two-step, Kandinsky, MDCP, max-class). |
| Components | **3, coupled.** C1 hypothesis-conditional forecaster; C2 calibrated membership posterior; C3 dual-weighted level-set calibration. | C3 needs f_c from C1 and p_c from C2. The dual weights correct for C2's miscalibration and C1's class-specific error. Trained objectives: the forecaster's pinball/L1 loss and the temperature NLL (2 < 3). |
| Metrics | Per-class coverage (point-pooled, as FLARE reports) and per-class object-averaged coverage; worst-class coverage; class-coverage gap CCG = mean_c abs(cov_c − (1−α)); worst under-coverage U; mean band length and hull width; Gneiting–Raftery interval score (pooled and class-averaged); class-averaged width; misclassified-object coverage; marginal coverage. Reported at 68% and 95%, at every forecast origin and every lead-time bin. | Coverage alone is gamed by wide bands. The interval score and width measure efficiency, and the class-averaged versions keep rare classes from being swamped. |
| Experiments beyond main and ablation | (1) Observation study: FLARE's per-class coverage versus classifier correctness and origin. (2) Synthetic check of Prop. 1/Thm 3 (known oracle). (3) Hyperparameter analysis on validation folds only (nested). (4) Efficiency frontier (coverage versus width). (5) Per-horizon and per-lead-time curves. (6) Finite-sample certificate (PRISM-cert) and object-level unit. (7) Robustness: classifier degradation (temperature, noise), calibration-set size, class-prior shift (PLAsTiCC train → DDF). (8) Runtime. (9) Paired object-level bootstrap significance. (10) Case studies (SLSN, CV light curves). (11) Optional: calibrating the next-observation closure probability. | These are what a top-venue conformal paper needs: validity, efficiency, the cause of the effect, sensitivity, and transfer. |
| Venue / template | ICML 2027, `icml2026.sty` (latest public style), 8 pages + references + unlimited appendix, Impact Statement. | The method, theory and benchmarks form an ML paper. The ZTF application gives the motivating case. |

## Core claims (to be tested; wording is provisional until the full runs)

1. On every main benchmark, PRISM's worst-class coverage is within sampling error of nominal at 68% and 95%, at every
   forecast origin. Membership-blind and probability-conditioned baselines under-cover at least one class.
2. Among methods whose worst-class coverage is at or near nominal, PRISM has the smallest band length and class-averaged
   interval score. It is narrower than two-step CP, max-class CP and MDCP, which ignore or under-use membership.
3. Each component matters. C1 (class-specific centres and scales), C2 (calibrated membership) and C3 (dual weights
   versus mixture or prior weights) each improve validity or efficiency.
4. Theory. (Prop. 1) The oracle band under per-class constraints is a level set of Σ_c v_c p_c f_c with constant dual weights.
   (Prop. 2) Any calibration based only on X-measurable weights has a class-c coverage error equal to a conditional
   covariance; this explains FLARE, GCC-p and mixture CP. (Thm 3) For fixed v, PRISM's min-threshold rule is valid for
   every class in finite samples; PRISM-cert attains this exactly. With in-sample v the error is
   O(sqrt(K log N / n_c)).

## Method summary

See `prism/methods.py::prism_dual` and the paper, Section 3. One cross-fit per benchmark and origin gives out-of-fold
classifier probabilities P, pooled forecasts and hypothesis-conditional forecasts (`scripts/crossfit.py`). Calibration
and evaluation are leave-one-fold-out over the five folds (`scripts/evaluate.py`).

## Key decisions

- MDCP is the closest prior work. Its Theorem 7 (appendix) states the marginal per-source optimum. Its algorithm
  targets x-conditional uniform coverage, treats the covariate density ratio as constant, and certifies by max-p.
  It is implemented faithfully (dual ERM with Adam, max-p) and given our forecaster and the posterior as its basis.
- Kandinsky's fractional-membership guarantee requires the true P(Z | X, Y). We give it a learned multinomial posterior
  on (classifier output, residuals under each hypothesis). PRISM's validity does not depend on posterior correctness.
- Same folds as FLARE: StratifiedKFold(5, shuffle, seed 42) over the BTS pool. Other benchmarks use the same rule.
- Unfavourable results (any benchmark, origin or level where PRISM loses) go to the appendix, unhidden.
