# Design log (ARIS: every design change with its evidence; nothing deleted)

All numbers: leave-one-fold-out on the development data indicated; recorded before the full runs.

1. **v0 union design (PRISM-union, kept as "GAMUT-union (early design)")**: conformal class set + union of class
   intervals + per-class CRC. ZTF h5 68%: marginal 0.788, width 1.49 (FLARE 1.07). Over-covers; superseded.
2. **v1 dual-weighted level set, point-level thresholds, in-sample dual**: ZTF h5 68% worst 0.667, width 1.026.
3. **v2 + temperature (C2) + object-level CRC + (n_c+1) correction**: ZTF h5 68% worst 0.677, CCG 0.004,
   width 1.021; 95% worst 0.925. Temperature: cls-IS 2.435 -> 2.419 (68%), 4.341 -> 4.277 (95%).
4. **Evaluation bug fixed (2026-11-22)**: empty bands were dropped from the metrics instead of counted as misses.
   Affected MDCP (6.7% empty at ZTF h5) and the global-threshold level set on the synthetic benchmark (10% empty).
   All results computed before the fix are archived in results/_archive_pilot and not used.
5. **Rejected: HPD-mass level** (band = highest-density region of the reweighted mixture with calibrated mass).
   Synthetic 68%: common class over-covered (0.97), width 2.75 vs 1.52 for the level set. A mass level cannot
   shrink bands where the rare class is implausible. Code removed; recorded here.
6. **Absolute length vs scale-normalized length**: absolute-length level sets abandon high-variance points (empty
   bands, late horizons). ZTF h5 68%: CV coverage at late lead bins -0.39 below nominal. Scale normalization
   (minimize E|C(X)|/sbar(X), sbar = posterior-mean scale; reduces to normalized-residual CP for K = 1): late-bin
   deviations <= 0.13. Adopted.
7. **Lead-bin Mondrian cells** (per-bin dual weights and threshold, as FLARE's cells): ZTF h5 68% worst 0.672,
   CCG 0.008, width 1.113 (FLARE 1.070, two-step 1.530); mean |cell deviation| 0.019 (FLARE 0.056); 95% worst 0.941,
   width 2.761 (FLARE 2.525, two-step 3.790). Synthetic: all cells within 0.01. Adopted as the final design;
   items 6-7 are reported as C3 ablations ("w/o scale normalization", "w/o lead-bin cells").
8. **Leave-one-object-out residual laws (adopted, GAMUT_VERSION v3-loo)**: with in-sample residual densities, small
   classes look better covered in calibration than on test (a calibration point's own kernel inflates its density).
   ZTF h20 68%: CV 0.652 pooled (object-averaged 0.716). Evaluating each calibration point's own-class density with
   its object's points removed: ZTF h20 68% all classes >= 0.678 (CV 0.678), width 0.677 vs 0.673; 95% all >= 0.951,
   width 1.868 vs 1.768. ZTF h5 68% all >= 0.676, width 1.121; 95% all >= 0.946, width 2.920 (FLARE 2.525, two-step 3.790).
9. **Hyperparameters chosen on validation folds only (nested, scripts/validate.py; ZTF h5 and h20)**:
   - The finite-sample (n_c+1) correction is necessary. Without it the worst class falls to 0.653 at 68% (h5, h20) and 0.928 at 95% (h5).
   - Pooling residual laws across bins is inconsistent: min_bin=500 is -4% width at h5 95% and +5% at h20 95%. The default min_bin=150 is kept.
   - Temperature (C2) narrows slightly on ZTF and more under miscalibration (synthetic kappa=0.5: -2 to 3%).
   - Bandwidth, pruning and dual iterations: worst-class changes <= 0.01.
   No default was changed after seeing test folds.
10. **Bandwidth multiplier x2 (v5), selected on validation folds**: at 68% no change. At 95% width -2.6% (h5) and
    -8.9% (h20), worst class 0.946 / 0.951 (validation). Silverman's rule under-smooths heavy-tailed residuals.
    All GAMUT entries recomputed with v5; baselines unchanged.
11. **Data-adaptive horizon cells (v6), selected on validation folds**: FLARE's lead-time bins are merged greedily
    (calibration labels and lead times only, never outcomes) until every class with >= ceil(2/alpha) calibration
    objects has that many objects in each cell. This is the analogue of FLARE's min-cell fallback.
    Nested validation:
    - PLAsTiCC-train h5: 68% width -7%, CCG 0.098 -> 0.080, worst 0.700; 95% width -14.5%, worst 0.946 -> 0.953.
    - ZTF h5: 68% unchanged; 95% width -9%, worst 0.946 -> 0.944.
    Motivated by PLAsTiCC-train test results (GAMUT 95% wider than MDCP), but decided on validation folds only.
    Recorded here as a possible source of optimism.
12. **Tried, not adopted: the class-inclusion cutoff and hierarchical pooling** (after ECG5000 h20 95% test: GAMUT
    2.277 vs two-step 2.126; class 5 has ~19 calibration objects, so its object-level CRC budget alpha(n+1)-1 is 0 and its
    threshold covers every calibration point):
    - Cutoff ceil(2/alpha) instead of 1/alpha-1 (validation): ECG identical; PLAsTiCC-train 95% width -30% but worst
      0.953 -> 0.882 (small classes lose protection). Rejected.
    - Hierarchical thresholds (per-bin for big classes, one pooled threshold for small ones), ECG 95% (test fold check):
      width 3.62, worse. With ~20 objects an object-level 95% guarantee forces near-total coverage, whether per bin or
      pooled.
    The default stays v6. The ECG5000 95% result is reported as unfavourable, with this explanation. The code keeps
    `incl` and `cells="hier"` as non-default options for the record.
13. **Measurement fix: adaptive evaluation grid (GAMUT v7; Kandinsky CP and MDCP grid g2)**: band hull and length
    were computed on a 400-point uniform grid. A band narrower than the grid spacing (point masses: 59% of
    ElectricDevices targets are exactly 0) appeared "empty" and was counted as a miss. GAMUT: 11.5% empty on
    ElectricDevices h20 68%, 8% on PLAsTiCC-train h5. Kandinsky: 23%. MDCP: 5%.
    The grid now adds 49 points within +-3 scales of every hypothesis centre, and length uses midpoint cell widths.
    The same function is used for all three grid-based methods, and all three are recomputed everywhere.
    ElectricDevices h20 68% GAMUT: worst 0.542 -> 0.680, width 0.768 (FLARE 0.783). The definition of the
    prediction set is unchanged; only its measurement changed.
14. **Residual-law support +-60 scales (v8; was +-12)**: the residual densities lived on [-12, 12] forecast scales, and
    points beyond got density ~0. ElectricDevices residuals are heavy-tailed: 2-4% of points beyond 12 scales in 5 of 7
    classes. At 95% this forced the threshold to ~0, so the band spanned the whole grid (width 6.76 on fold 0).
    With +-60: width 3.33, per-class coverage 0.94-0.97 (fold 0). This is a correction of an arbitrary truncation,
    not a tuned hyperparameter. All GAMUT entries are recomputed.
15. **Round-4 response (no change to GAMUT; v8 frozen)**: (a) new baselines MDCP + C1, Two-step CP + C1 and Max-class
    CP + C1 (`gamut/latent_baselines.py`); (b) strict three-way split at all five ZTF origins with MDCP, Kandinsky,
    CPTC and the +C1 baselines, and the object-level bootstrap of the main text (`scripts/strict_split.py`);
    (c) pre-registered held-out benchmark PLAsTiCC-WFD (test chunk 02), run once with frozen code
    (`refine-logs/EXPERIMENT_PLAN.md`, `scripts/heldout_wfd.sh`); (d) the object-level bootstrap is now seeded per
    (setting) and per (setting, method) for the cell test, so that its Monte-Carlo draws do not depend on the list of
    methods. Before this, adding the +C1 baselines shifted the random stream for later settings and moved Kandinsky
    CP's validity by one setting per level (75/60% -> 70/55%; the stable seeds give 70/55%); GAMUT's rates (100/90%) did
    not change. All validity numbers are recomputed with the stable seeds.
    (e) An obsolete one-off relabelling script for pre-v8 validation files was retired to `scripts/_obsolete/` before
    it could run on the v8 files.
16. **Closure probability (pre-registered, see EXPERIMENT_PLAN.md)**: latent-class mixture of next-point laws (C1
    forecaster + class-c empirical residuals, C2 posterior) + inductive Venn-Abers, against FLARE's deployed
    p_singleton on FLARE's own splits (calibrators on `val`, report on `test`). Implementation notes: the worker pool
    uses `spawn` (fork after LightGBM/OpenMP initialisation deadlocked); missing catalogue context is passed as NaN.
    Classes with fewer than 30 validation residuals (SLSN, AGN, TDE) use the pooled residual law; this rule was not in
    the registration (it was fixed in the script before the first run) and is disclosed in the paper. After the
    post-loop verification review, C1 was retrained on FLARE's `train` split only (scripts/closure_c1.py) so that no
    val/test object enters the score's training; the first run (cross-fitted C1) is kept as a sensitivity result.
    First run (cross-fitted C1): Brier -25% vs deployed, -19%/-20% vs deployed + Venn-Abers. Final run (C1 on `train`):
    Brier -27%/-19% vs deployed, -21%/-14% vs deployed + Venn-Abers, AUC 0.937->0.963 and 0.955->0.973, ECE on par
    with the recalibrated deployed score. A pooled control (scripts/closure_pooled.py) reaches Brier 0.080/0.064 vs
    ours 0.077/0.064: the gain is mainly the learned next-point law. Worse for SLSNe (both days) and AGN/TDE (day 10).
17. **Registered B1 paired IS bootstrap** (scripts/is_bootstrap.py, run after the second verification review):
    GAMUT vs the best valid baseline on class-averaged IS is significantly better in 3/20 (68%) and 5/20 (95%)
    settings and significantly worse in 13/20 and 14/20. Reported in the registered-outcomes table (tab:plan).
18. **GAMUT-Q (registered interval-score variant)**: central intervals of the dual-weighted mixture CDF, same C1-C3,
    in gamut/variants.py (gamut/methods.py kept byte-identical). Development (test-informed): valid 95%/100%, width
    1.62x/1.75x FLARE, IS better/worse than the best valid method in 3/13 and 2/17 settings. Confirmation on the untouched
    PLAsTiCC chunk 03 (frozen code): valid in 100%/33% of settings, IS better/worse 3/0 and 1/2 -> registered claim not
    met, not adopted. Bug found on the way: run_all.py re-evaluated unversioned GAMUT-named entries forever; fixed by
    stamping the version on variants in evaluate.py.
19. **Second held-out confirmation of GAMUT** on chunk 03: valid at every origin and level, bands 1.40x/1.58x FLARE vs
    1.83-2.01x/1.54-2.64x for the always-valid baselines; FLARE valid nowhere.
20. **Analyses of the remaining weaknesses** (no method change): IS against each always-valid baseline; validity on
    GAMUT's own merged cells (default 5.5/1.9 under-covered cells per setting, GAMUT-cert 0.8/0.0); closure against FLARE's
    pre-registered +-0.10 five-bin criterion (deployed 2/5 and 3/5 bins, ours without recalibration 5/5 and 4/5);
    rare-class paired bootstrap; class-specific residual laws for the closure study from within-train out-of-fold C1
    (post hoc; day-10 SLSN worse, not adopted); ElectricDevices sensitivity; day-50 per-class figure in the format of
    FLARE's forecast paper.
