# GAMUT — Class-conditional forecast bands under type uncertainty

**GAMUT** (*Guarantees for All Members Under Type uncertainty*) is one conformal calibration method for multi-horizon forecast bands that targets coverage **for every latent class**. The class is labelled in the calibration data but unknown at test time. The default cross-fitted protocol is evaluated empirically; a strict train/calibration/test protocol gives the same method a finite-sample guarantee for eligible class--horizon cells with enough certification objects. The motivating application is early light-curve forecasting of astronomical transients with
[FLARE](https://github.com/applecider-ml/flare) on the ZTF Bright Transient Survey (BTS). FLARE's bands reach nominal
coverage over the whole population, but they under-cover superluminous supernovae (SLSNe) and, at later horizons,
cataclysmic variables (CVs). Its conformal multiplier depends on the *predicted* class, which is often wrong in the
first days for rare objects, so the band is calibrated for the wrong population.

This repository contains the method, 13 external baselines (plus three of them given GAMUT's forecaster), six main
benchmarks, an appendix benchmark, two held-out PLAsTiCC chunks, a synthetic family, the reproducibility scripts and
both ICML-style manuscript versions: the double-blind submission (`paper/main.tex` / `paper/main.pdf`) and the full-information author preprint (`paper/main_authors.tex` / `paper/main_authors.pdf`).

---

## Contents

1. [Summary of findings](#1-summary-of-findings)
2. [Problem](#2-problem)
3. [Theory](#3-theory)
4. [Method](#4-method)
5. [Experimental design](#5-experimental-design)
6. [Results](#6-results)
7. [Additional analyses and scope](#7-additional-analyses-and-scope)
8. [Second open problem: the closure probability](#8-second-open-problem-the-closure-probability)
9. [Reproducing the results](#9-reproducing-the-results)
10. [Repository layout](#10-repository-layout)
11. [Citation and acknowledgements](#11-citation-and-acknowledgements)

---

## 1. Summary of findings

- **Why calibrating by class probabilities is not enough (Prop. 1).** Calibrating by the predicted class, or by
  reweighting with the classifier's probabilities, leaves a class-coverage error equal to a conditional covariance
  between membership and miscoverage, plus the posterior's own error. On ZTF, FLARE's 68% band covers day-5 SLSNe at
  **0.53**. Misclassified objects are covered at 0.61 against 0.70 for correctly classified ones. Five
  probability-conditioned conformal methods raise the worst class by at most 0.069.
- **What the optimal band looks like (Prop. 2).** The band of smallest scale-weighted length with one coverage
  constraint per class and horizon is a level set of a **dual-weighted mixture of class-conditional predictive
  densities**.
- **What GAMUT does.** It estimates that band with three coupled components:
  - **C1**, a hypothesis-conditional forecaster;
  - **C2**, a calibrated membership posterior;
  - **C3**, a dual fixed point that equalises per-class conformal risk-control thresholds.
- **Strict-split guarantee.** The same GAMUT calibration rule is valid per class and horizon cell for every class with
  enough certification objects when nuisance fits and thresholds are separated by a strict train/calibration/test
  split. The guarantee rests on the calibration labels, not on a correct posterior.
- **Six main benchmarks, 20 settings per level.** "Valid" means no class is significantly under-covered under an
  object-level bootstrap test. The default GAMUT is valid in **100%** of settings at 68% and **90%** at 95%, with
  bands **1.09×** and **1.36×** FLARE's. The always-valid baselines (two-step, MDCP, max-class) need 1.48–1.62× and
  1.49–2.61×. GAMUT is on average **27% (68%) and 20% (95%) shorter** than them (only 8% shorter than MDCP at 95%,
  and wider than MDCP in 7/20 settings there) and 29% shorter than their versions given GAMUT's forecaster.
- **On ZTF**, day-5 SLSN coverage at 68% rises from 0.53 to **0.71**, with bands only **1.05×** FLARE's.
- **Strict split on ZTF**, at all five origins (the premise of the theorem):
  - GAMUT has no under-covered class at either level, with bands **1.08×** (68%) and 1.70× (95%) FLARE's.
  - The always-valid baselines need 1.84–2.01× and 1.88–2.18×.
  - FLARE under-covers a class at every origin.
- **Pre-registered held-out set** (PLAsTiCC test chunk 02, wide-fast-deep; 20,763 raw objects, 20,760 with a future point at the earliest origin; frozen code, run once):
  - GAMUT is valid at all three origins at both levels, with bands 1.51× and 1.67× FLARE's.
  - The always-valid baselines need 1.84–2.02× and 1.49–1.96×; MDCP is narrower than GAMUT at 95%.
  - FLARE's worst class falls to 0.27.
- **Second untouched held-out set** (PLAsTiCC test chunk 03): GAMUT is again valid at every origin and level, with bands
  1.40× and 1.58× FLARE's (always-valid baselines 1.83–2.01× and 1.54–2.64×); FLARE is valid nowhere.
- **Closure probability** (FLARE's second open problem): simulating the next observation from a learned next-point
  law (GAMUT's C1 with empirical residuals, mixed by C2) instead of a fixed-scatter Gaussian, then adding a Venn–Abers
  layer, lowers the test Brier score:
  - by **21%** (day 5) and **14%** (day 10) against FLARE's score with the same Venn–Abers layer;
  - by 27% and 19% against FLARE's score as shipped.
  
  AUC rises from 0.937 to 0.963 and from 0.955 to 0.973. On the ±0.10 five-bin tolerance pre-registered in FLARE's
  forecast paper, the deployed score passes 2/5 (day 5) and 3/5 (day 10) bins; ours passes 5/5 and 4/5 **without any
  recalibration**. A pooled version of the same learned law does almost as well (Brier 0.080 and 0.064), so the gain
  comes mainly from the learned law, not from class conditioning.

Every unfavourable result is reported in the paper's appendix and summarised in [Section 7](#7-where-gamut-does-not-win).

---

## 2. Problem

An object (a transient) has a prefix $x_0$: its light curve up to the forecast origin, summarised as features. It
also has a latent class $Z\in[K]$ and future measurements $Y_j$ at lead times $\ell_j$. Lead times are grouped into
horizon cells $b$, which are FLARE's lead-time bins, merged only when data are scarce. Objects are the exchangeable
units, so the target is the **object-level coverage per class and horizon cell**:

$$
\mathrm{cov}_{c,b}(C)=\mathbb E\Big[\tfrac{1}{m_b}\textstyle\sum_{j:\,b(\ell_j)=b}\mathbf 1\{Y_j\in C(x_j)\}\ \Big|\ Z=c,\ m_b\ge 1\Big]\ \ge\ 1-\alpha
\quad\text{for every } c, b .
$$

The calibration set holds $n$ objects with known *true* classes, for example from spectroscopy. At test time only a
classifier posterior $\hat p(x_0)\in\Delta_K$ is available. This places the problem outside group-conditional
conformal prediction, which needs the group at test time, and outside class-conditional prediction sets for
classification.

## 3. Theory

- **Prop. 1 (coverage gap).** For any band, $\mathbb P(M=1\mid Z=c)=\mathbb E[p_c(X)M]/\pi_c+\mathbb E[\mathrm{Cov}(\mathbf 1\{Z=c\},M\mid X)]/\pi_c$,
  where $M$ is the miss indicator.
  - A calibration that enforces $\mathbb E[\hat p_c(X)(M-\alpha)]=0$ leaves the covariance term plus a
    posterior-error term. The conditional-guarantee method of Gibbs, Cherian and Candès, with $\hat p_c$ in its
    function class, is an example.
  - Mondrian calibration by the predicted class leaves the covariance given the predicted class.
  - The covariance is positive for rare classes with broad light curves.
- **Prop. 2 (oracle band).** Weight length by $1/\bar\sigma(x)$ and impose one constraint per class in a horizon cell.
  Then the shortest feasible band is
  $C^\star(x)=\{y:\ \bar\sigma(x)\sum_c v^\star_c\,p_c(x)\,f_c(y\mid x)\ge 1\}$,
  with constant dual weights $v^\star_c=\lambda^\star_c/\pi_c$. The proof uses LP duality via a Slater point and
  weak-* compactness. This is the latent-class analogue of the marginal per-source optimum in MDCP and of
  class-conditional set-valued classification.
- **the strict-split guarantee (finite-sample validity of strict-split GAMUT).** Assume base models are trained on objects disjoint from the
  calibration and test objects. Split the calibration objects into halves $A$ and $B$. Fit everything except the
  thresholds on $A$; compute per-class conformal risk-control thresholds on $B$. Then every class $c$ with
  $n^B_{c,b}\ge 1/\alpha-1$ satisfies $\mathrm{cov}_{c,b}\ge 1-\alpha$ in every horizon cell $b$. The result holds
  conditionally on $A$ and on the classes and cell memberships of $B$, whatever the quality of the posterior or the
  forecaster.
- **A uniform-convergence bound** for in-sample weights is given in the appendix. It is not informative at these
  class sizes (slack about 2.0 at $n_c=150$), so the default GAMUT is validated empirically.

## 4. Method

| Component | What it does | Why it is needed |
|---|---|---|
| **C1 — hypothesis-conditional forecaster** | One LightGBM median/quantile model with the class one-hot appended, queried under **every** class hypothesis. Per (class, lead-bin) kernel residual laws on standardised residuals (support ±60 scales, 2× Silverman bandwidth), scored **leave-one-object-out** for calibration points. | Supplies $\hat f_c(y\mid x)$ for every class, including the classes the classifier rejects. |
| **C2 — calibrated membership** | Temperature scaling of the classifier posterior on the calibration objects. | Supplies $\hat p_c(x)$ in the oracle form. |
| **C3 — dual-weighted level-set calibration** | In each horizon cell, a damped multiplicative fixed point finds weights $v$ that equalise the per-class object-level conformal risk-control thresholds $t_c(v)$. The band uses $\hat t=\min_c t_c(v)$ over classes with $n_c\ge 1/\alpha-1$. Horizon cells are FLARE's bins, merged until every class with at least $\lceil 2/\alpha\rceil$ objects has that many per cell. | Estimates the dual weights $v^\star$ of Prop. 2 and gives per-class validity. |

The band is $C(x)=\{y:\ \bar\sigma(x)\sum_c v_{b(x),c}\,\hat p_c(x)\,\hat f_c(y\mid x)\ge \hat t_{b(x)}\}$. It can be
a union of intervals when the class forecasts disagree. Under the strict protocol, nuisance quantities are fitted on
half $A$ and thresholds on half $B$; this is the certified strict-split evaluation of GAMUT. On ZTF, calibration and
prediction take about 8 s per fold on one CPU core. The implementation is `gamut.methods.gamut` (`crc=True` selects the
strict protocol).

## 5. Experimental design

**Benchmarks.** "Min" is the size of the smallest class.

| Benchmark | Domain | Objects | K | Min | Future points | Origins |
|---|---|---|---|---|---|---|
| ZTF-BTS (FLARE's public benchmark and folds) | ZTF transients | 10,492 | 6 | 68 | 273,908 | 5, 10, 15, 20, 30 d |
| PLAsTiCC-train | LSST simulation | 2,045 | 11 | 36 | 21,461 | 5, 15, 30 d |
| PLAsTiCC-DDF (test chunk 01) | LSST simulation, deep fields | 8,681 | 9 | 42 | 74,139 | 5, 15, 30 d |
| UCR StarLightCurves | stellar light curves | 9,236 | 3 | 1,329 | 221,664 | 20/40/60% prefix |
| UCR ECG5000 | ECG | 5,000 | 5 | 24 | 120,000 | 20/40/60% prefix |
| UCR ElectricDevices | electricity | 16,637 | 7 | 1,252 | 399,288 | 20/40/60% prefix |
| UCR Wafer (appendix) | semiconductor | 7,164 | 2 | 762 | 171,936 | 20% prefix |
| **PLAsTiCC-WFD (pre-registered held-out, test chunk 02)** | LSST simulation, wide-fast-deep | 20,760 | 12 | 30 | 115,532 | 5, 15, 30 d |
| Synthetic family κ ∈ {0.5, 1, 2} | known posterior and class laws | 6,000 | 3 | ~300 | 60,000 | — |

**Baselines.** 13 literature baselines. Where a baseline needed a component missing in this setting, it was given ours.
- *Deployed:* FLARE's Mondrian bands by predicted class.
- *Membership-blind:* split CP, CQR, DCP.
- *Probability-conditioned:*
  - localized CP in posterior space;
  - the label-shift mixture of class-wise score laws;
  - GCC (Gibbs–Cherian–Candès) with posterior features;
  - learned-partition Mondrian CP;
  - CPTC.
- *Latent-class-valid:* two-step CP, Kandinsky CP, MDCP, max-class CP.
- *Same forecaster as GAMUT:* MDCP, two-step CP and max-class CP are also run with GAMUT's C1 ("+ C1"), to separate
  calibration from modelling.
- *Reference:* a true-class Mondrian band, which needs the test label.

**Metrics.**
- Per-class coverage, pooled over points (FLARE's metric) and object-level per (class, lead-time bin).
- Worst-class coverage and class-coverage gap (CCG).
- Mean band length; interval score and its class average.
- **Validity:** no class significantly under-covered, i.e. the upper end of a 95% object-level bootstrap interval
  (500 resamples, seeded per setting) is not below nominal. The pre-registered fixed-tolerance rule is also reported.
- For the closure probability: ECE, Brier, log loss, AUC, calibration slope and per-class reliability.

**Protocol.** Leave-one-fold-out cross-conformal over five stratified folds, which is FLARE's protocol: every method
uses the same cross-fitted classifier and forecaster outputs, and only the calibration step differs. Hyperparameters
are selected on nested validation folds only.

**Experiments beyond the main comparison.**

| Experiment | Purpose |
|---|---|
| Observation study (FLARE on ZTF; probability-conditioned methods) | Measure the failure that Prop. 1 explains |
| Synthetic family with known laws | Check Prop. 1 against the truth; distance to the oracle band of Prop. 2 |
| Ablations (12 variants) | Contribution of C1, C2, C3, leave-one-object-out laws, horizon cells, scale normalisation, units |
| Hyperparameter analysis on validation folds | Sensitivity of pooling size, bandwidth, pruning and dual iterations |
| Strict three-way split, ZTF, all origins | The only protocol that meets the strict-split guarantee's premise |
| Pre-registered held-out PLAsTiCC-WFD | Guard against design choices that followed test results |
| Runtime | Cost per fold |
| Closure probability (pre-registered) | FLARE's second open problem ([Section 8](#8-second-open-problem-the-closure-probability)) |

## 6. Results

**Main table** (six benchmarks, all origins, 20 settings per level). *Valid*: % of settings with no class
significantly under-covered. *Worst*: mean worst-class coverage. *Cells*: mean number of significantly under-covered
(class, lead-time bin) cells per setting. *Width*: band length relative to FLARE's.

| Method | 68%: Valid | Worst | CCG | Cells | Width | 95%: Valid | Worst | CCG | Cells | Width |
|---|---|---|---|---|---|---|---|---|---|---|
| FLARE | 5 | 0.556 | 0.042 | 10.2 | 1.00 | 10 | 0.879 | 0.021 | 7.0 | 1.00 |
| Split CP | 5 | 0.519 | 0.074 | 14.8 | 0.95 | 5 | 0.870 | 0.029 | 11.9 | 0.95 |
| CQR | 5 | 0.515 | 0.079 | 20.5 | 1.01 | 0 | 0.851 | 0.039 | 14.6 | 0.93 |
| DCP | 0 | 0.532 | 0.069 | 17.1 | 1.07 | 0 | 0.880 | 0.026 | 10.3 | 1.11 |
| Localized CP | 10 | 0.534 | 0.065 | 13.2 | 0.96 | 5 | 0.876 | 0.026 | 9.8 | 0.96 |
| Label-shift mixture | 30 | 0.575 | 0.037 | 6.5 | 1.01 | 30 | 0.894 | 0.017 | 5.0 | 1.00 |
| GCC (p) | 20 | 0.563 | 0.042 | 11.9 | 1.00 | 15 | 0.879 | 0.021 | 8.9 | 0.98 |
| Learned partition | 30 | 0.563 | 0.039 | 9.2 | 1.01 | 25 | 0.883 | 0.020 | 6.8 | 1.00 |
| CPTC | 35 | 0.585 | 0.038 | 5.8 | 1.01 | 80 | 0.922 | 0.017 | 1.1 | 1.17 |
| Two-step CP | 100 | 0.700 | 0.104 | 0.0 | 1.48 | 100 | 0.956 | 0.025 | 0.0 | 1.84 |
| Kandinsky CP | 70 | 0.615 | 0.021 | 7.5 | 1.07 | 55 | 0.917 | 0.010 | 4.8 | 1.10 |
| MDCP | 100 | 0.705 | 0.130 | 1.8 | 1.62 | 100 | 0.948 | 0.028 | 1.9 | 1.49 |
| Max-class CP | 100 | 0.704 | 0.121 | 0.0 | 1.55 | 100 | 0.958 | 0.030 | 0.0 | 2.61 |
| Two-step CP + C1 | 100 | 0.734 | 0.135 | 0.0 | 1.70 | 100 | 0.962 | 0.029 | 0.0 | 1.75 |
| MDCP + C1 | 100 | 0.702 | 0.141 | 2.5 | 1.73 | 100 | 0.949 | 0.030 | 1.6 | 1.54 |
| Max-class CP + C1 | 100 | 0.751 | 0.174 | 0.0 | 1.96 | 100 | 0.968 | 0.036 | 0.0 | 2.22 |
| **GAMUT (ours)** | **100** | 0.671 | **0.014** | 0.2 | **1.09** | 90 | 0.949 | **0.012** | 0.8 | **1.36** |
| GAMUT (split calibration) | 100 | 0.745 | 0.096 | 0.0 | 1.47 | 100 | 0.968 | 0.028 | 0.1 | 1.62 |
| True-class Mondrian (reference) | 85 | 0.633 | 0.008 | 1.6 | 1.00 | 90 | 0.926 | 0.004 | 0.2 | 1.02 |

**Strict train/calibration/test split** (ZTF, five origins). Valid = % of origins with no class significantly
under-covered.

| Method | 68%: Valid | Width | 95%: Valid | Width |
|---|---|---|---|---|
| FLARE | 0 | 1.00 | 0 | 1.00 |
| Two-step / max-class / MDCP | 100 | 1.84–2.01 | 100 | 1.88–2.18 |
| **GAMUT** | **100** | **1.08** | **100** | 1.70 |
| GAMUT (strict split; certificate) | 100 | 1.70 | 100 | 1.27* |

\* At 95%, half of one calibration fold leaves SLSN, AGN, TDE and CV below the $1/\alpha-1$ certification objects a
threshold needs. strict-split GAMUT's guarantee then covers only the two supernova classes; the others are covered
empirically but not certified.

**Pre-registered held-out set** (PLAsTiCC-WFD, 3 origins; 20,763 objects pass the quality cut, 20,760 of them have
a future point at the earliest origin). FLARE: valid 0% at both levels, worst class 0.266.
GAMUT: valid **100%** at both levels, width **1.51×** and **1.67×**. Two-step, MDCP and max-class CP: valid 100%,
width 1.84–2.02× and 1.49–1.96×.

**Day-50 coverage per class** (the format of FLARE's forecast paper, Fig. 2; `paper/figures/day50.pdf`):
FLARE's 68% band covers about half of the SLSN day-50 measurements, as that paper reports; GAMUT covers about 0.70.

**Synthetic family.** Membership-blind and probability-conditioned methods cover the rare class at 0.28–0.36 (68%).
Prop. 1 predicts the measured class coverage to within 0.026. GAMUT is only **2.3%** (68%) and **7.5%** (95%) longer
than the oracle band of Prop. 2; the latent-class-valid baselines are 69–73% and 22–40% longer.

**Ablations** (relative to full GAMUT, six benchmarks).

| Variant | Effect |
|---|---|
| Uniform posterior (no C2) | Bands +45% longer |
| Argmax posterior (no C2) | Bands +36% longer |
| Posterior mixture without dual weights (no C3) | Valid in 0% of settings at 68% (worst class 0.516) |
| Prior-ratio weights (no fixed point) | Bands +31% longer; over-covers the common classes |
| No leave-one-object-out laws | Validity at 95% drops to 75% |
| One threshold for all horizons | A (class, horizon) cell falls to 0.42 on ZTF |
| No C1 (pooled forecaster) | +4% / +3% on real data (validity at 95% 90% → 85%); +16% / +33% on synthetic data |
| No temperature scaling | +1% on real data; matters for over-confident posteriors (synthetic κ = 2: +9% / +15%) |
| Fixed lead-time bins (no merging) | +1% / +3% on test folds and fewer under-covered cells; selected against on validation folds (up to 18% longer there) |

**Hyperparameters** (validation folds only). Pooling size, bandwidth, pruning and the number of dual iterations
change the worst class by ≤ 0.017 and the width by ≤ 10%. Stopping the fixed point after 5 iterations raises the CCG
from 0.019 to 0.082 on ECG5000.

## 7. Additional analyses and scope

The appendix reports the ablations, strict-split results, interval-score and per-cell analyses, small-class behavior,
empty-band rates, and the preregistered endpoints. These analyses define the operating scope of the default
cross-fitted protocol and motivate the certified strict-split evaluation. The paper keeps the full tables and
protocol details for readers who want to inspect each setting.

## 8. Second open problem: the closure probability

FLARE reports $p_{\rm singleton}$, the probability that one more $g$/$r$ detection makes its 90% classification set
a singleton. It simulates 16 next-night magnitudes from the predicted class's forecast with a fixed scatter. The law
of the next point depends on the latent class, so we use

$$
p_{\rm close}(x)=\textstyle\sum_c \hat p_c(x)\ \mathbb P\big(|\Gamma(x\oplus Y)|=1\ \big|\ Y\sim\hat f_c(\cdot\mid x)\big),
$$

followed by an inductive Venn–Abers layer. Here $\hat f_c$ is GAMUT's C1 with class-$c$ empirical residuals and
$\hat p$ is C2.

**Protocol.** The study was pre-registered. All models are trained on FLARE's `train` split, calibrators are fitted
on `val`, and results are reported on `test`. The outcome is whether the deployed set becomes a singleton after the
object's *real* next detection.

| Day | Score | ECE | Brier | Log loss | AUC |
|---|---|---|---|---|---|
| 5 | FLARE p_singleton (as shipped) | 0.071 | 0.105 | 0.403 | 0.937 |
| 5 | FLARE + Venn–Abers | 0.023 | 0.098 | 0.319 | 0.937 |
| 5 | **Class-conditional + Venn–Abers (ours)** | 0.023 | **0.077** | **0.247** | **0.963** |
| 5 | Pooled learned law + Venn–Abers (control) | 0.025 | 0.080 | 0.256 | 0.959 |
| 10 | FLARE p_singleton (as shipped) | 0.054 | 0.078 | 0.356 | 0.955 |
| 10 | FLARE + Venn–Abers | 0.017 | 0.074 | 0.254 | 0.954 |
| 10 | Pooled learned law + Venn–Abers (control) | 0.019 | 0.064 | 0.213 | 0.968 |
| 10 | **Class-conditional + Venn–Abers (ours)** | 0.017 | **0.064** | **0.203** | **0.973** |

The paired bootstrap intervals of the Brier difference against FLARE + Venn–Abers exclude zero:
$[-0.028,-0.013]$ at day 5 and $[-0.019,-0.002]$ at day 10.

What the study does **not** show:
- ECE is not significantly lower than that of the recalibrated FLARE score. That registered claim is not met.
- The gain comes mainly from the learned next-point law: the pooled control is almost as good. Class conditioning
  adds AUC 0.004–0.005 and at most 0.003 Brier; mixing over classes adds 0.001 Brier over the argmax class alone.
- Per rare class (7–20 test rows each), the differences are not significant except SLSNe at day 10 with Venn–Abers
  (+0.043 Brier, interval [+0.009, +0.089]). Class-specific residual laws (post hoc) did not fix this.

## 9. Reproducing the results

**Requirements.** Python ≥ 3.11, plus `lightgbm`, `scikit-learn`, `scipy`, `pandas`, `pyarrow` and `matplotlib`.
You also need a checkout of [FLARE](https://github.com/applecider-ml/flare), which provides the ZTF features,
classifier and deployed models, installed with `pip install -e`. The paper needs a TeX Live installation.

```bash
export FLARE_REPO=/path/to/flare                 # FLARE checkout (default: ../flare)
export FLARE_PAPER_DATA=/path/to/paper_data      # FLARE paper data with data/splits and event files (default: ../paper_data)
export PYTHONPATH=$PWD:$FLARE_REPO
pip install -e $FLARE_REPO lightgbm scikit-learn scipy pandas pyarrow matplotlib

# 1. benchmarks (data are not redistributed: ZTF BTS via FLARE, PLAsTiCC from Zenodo 2539456, UCR archive)
python -m benchmarks.ztf --paper-data $FLARE_PAPER_DATA
python -m benchmarks.plasticc --which train      # also: --which ddf, --which wfd (held-out)
for d in StarLightCurves ECG5000 ElectricDevices Wafer; do python -m benchmarks.ucr --dataset $d; done
python -m benchmarks.synthetic --kappa 0.5 1 2

# 2. cross-fitted classifier / forecaster / hypothesis-conditional forecaster outputs
python scripts/crossfit.py --bench ztf --h 5 --lr 0.05 --cap 24   # every benchmark and origin; see scripts/queue_crossfits.sh

# 3. every method, ablation and baseline; validation; bootstrap; synthetic oracle
python scripts/run_all.py
python scripts/validate.py --bench ztf --h 5
python scripts/bootstrap.py && python scripts/synthetic_oracle.py

# 4. strict split, held-out set, closure probability
python scripts/strict_split.py --bench ztf --h 5                  # and 10, 15, 20, 30
bash scripts/heldout_wfd.sh
python scripts/closure_c1.py --h 5 && python scripts/closure.py --h 5   # and --h 10
python scripts/closure_eval.py

# 5. every table, macro and figure of the paper, then both PDFs (double-blind + preprint)
python scripts/make_tables.py && python scripts/make_figures.py
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex main_authors && bibtex main_authors && pdflatex main_authors && pdflatex main_authors
```

**Compute.** Everything was run on 4 CPU cores without a GPU:
- a cross-fit takes about 20 min per (benchmark, origin) on ZTF;
- the full evaluation of all methods takes a few minutes to about 40 min per setting (Kandinsky CP and MDCP dominate);
- each strict-split origin takes about 30 min.

**Determinism.**
- Folds come from `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Bootstrap draws are seeded per setting (and per method for the cell test).
- Strict-split rotations, MDCP and Kandinsky use fixed seeds.
- All numbers in the paper are generated from `results/` by `scripts/make_tables.py` and `scripts/closure_eval.py`,
  into `paper/numbers.tex`, `paper/closure_numbers.tex` and `results/macros.json`.

## 10. Repository layout

| Path | Content |
|---|---|
| `gamut/methods.py` | GAMUT (`gamut`, `crc=True` for strict-split GAMUT), the ablation registry, and the conformal baselines: FLARE bands, split CP, CQR, DCP, localized CP, label-shift mixture, GCC, learned partition |
| `gamut/latent_baselines.py` | Two-step CP, max-class CP, CPTC, Kandinsky CP, MDCP, and their "+ C1" variants |
| `gamut/data.py`, `gamut/metrics.py` | Task arrays (point level, object ids, lead bins) and every metric |
| `benchmarks/` | Builders for ZTF-BTS, PLAsTiCC (train / DDF / WFD), UCR and the synthetic family; the registry of classifier recipes |
| `scripts/crossfit.py` | Out-of-fold classifier, base forecaster and hypothesis-conditional forecaster |
| `scripts/evaluate.py`, `scripts/run_all.py` | Leave-one-fold-out evaluation of every method and ablation (lock-file work queue) |
| `scripts/validate.py` | Nested hyperparameter analysis on validation folds only |
| `scripts/bootstrap.py` | Object-level bootstrap validity per class and per (class, lead-time bin) |
| `scripts/synthetic_oracle.py` | Oracle band of Prop. 2 and the Prop. 1 check on synthetic data |
| `scripts/strict_split.py` | Strict train/calibration/test split at every ZTF origin |
| `scripts/heldout_wfd.sh` | The pre-registered held-out evaluation |
| `scripts/closure_c1.py`, `scripts/closure.py`, `scripts/closure_eval.py` | Closure-probability study |
| `scripts/make_tables.py`, `scripts/make_figures.py` | Every table, macro and figure of the paper |
| `results/` | All raw results: per-setting JSON, logs, bootstrap, strict split, held-out set, closure, macros |
| `paper/` | ICML 2026 manuscripts: double-blind review (`main.tex`, `main.pdf`) and full-information preprint (`main_authors.tex`, `main_authors.pdf`), plus shared `sections/`, `tables/`, and `figures/` |


## 11. Citation and acknowledgements

```bibtex
@misc{yan2026gamut,
  title  = {GAMUT: Class-Conditional Conformal Forecast Bands Under Latent Type Uncertainty},
  author = {Yan, Weiping and Sasli, Argyro},
  year   = {2026},
  note   = {Preprint and code: https://github.com/appleweiping/GAMUT}
}
```

The ZTF benchmark, features, classifier and deployed forecaster come from FLARE (Sasli et al., 2026) and the ZTF
Bright Transient Survey. PLAsTiCC is the unblinded LSST simulation (Kessler et al., 2019; Zenodo 2539456). The UCR
time-series archive is by Dau et al. (2019). No data are redistributed in this repository.
