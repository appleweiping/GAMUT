# Review round 3: full-paper claim audit (ICML AC style)

review_independence: same-family · fresh subagent · score 4/10 · confidence 4/5 · verdict: not ready

All numbers trace to the current results: GAMUT v8, Kandinsky/MDCP grid g2. Macros and tables match.
Problems found and responses:

| Issue | Response |
|---|---|
| Intro: "only method with no under-covered class on ZTF and synthetic" is false; two-step, MDCP and max-class also achieve it | Rewritten: those methods also achieve it, with bands 40-80% longer |
| "12 published methods" (13 evaluated) | Fixed to 13 |
| Abstract width claim omits the cases where GAMUT is wider | Abstract names the three baselines; main text states "wider than MDCP in k/20 settings at 95%" |
| Validity criterion changed from the pre-registered tolerance to the bootstrap, undisclosed (GAMUT 85% vs 100% at 68%) | Disclosed in Metrics and in the main results; appendix table reports both rules for every method |
| Ablation table mixes criteria; "every component matters" overstated; temperature ~0 on real data; C1 ~3% | Ablations bootstrapped with the same rule; paragraph rewritten with measured effects (largest from C2 membership and C3 dual; C1 3% real / 16-35% synthetic; temperature matters only for an over-confident classifier) |
| Synthetic kappa test is degenerate (pure temperature distortion) | Stated explicitly |
| Hyperparameter analysis stale (pre-v8) and one setting only | Re-run on v8 for ZTF h5, h20, PLAsTiCC-train h5, ECG5000 h40, both levels |
| Guarantee-matched comparison missing (GAMUT-cert ~ two-step at 68%, wider than MDCP at 95%) | Stated in the main text, with macros |
| CPTC at 95% valid in 80% with shorter bands, not mentioned | Stated |
| Small-class benchmarks description imprecise | Corrected (PLAsTiCC-train vs MDCP; ECG5000 vs all valid baselines) |
| Cert "conservative especially at 95%" wrong | Restated with mean worst class at both levels |
| Strict split: 95% width 1.52x FLARE not stated; single origin | Stated |
| Iteration cap 34% hides 90-100% on small-class benchmarks | Stated |
| Appendix: Kandinsky/CPTC invalid fractions wrong; "two alternatives on validation folds" inaccurate; hull claim unsupported | Corrected |
| Test-informed design changes undisclosed | New appendix section "Design history and test-set exposure" |
| Fig. 1 "reproduces" | "our re-implementation", with the differences listed |
| MDCP "given ours": it uses Gaussian densities; two-step/max-class use the pooled forecaster; QR subsampling | Stated in Baselines |
| "Oracle" naming | Renamed "true-class Mondrian (reference)", not an optimality bound |
| Misclassified-object coverage promised but not reported | Removed from metrics |
| UCR "five bins" (six), object counts, duplicate label | Fixed |
| Not addressed (time/compute): strict split at more origins with bootstrap; MDCP with GAMUT's residual laws; two-step with C1; label-noise robustness | Listed as open in the final report |
