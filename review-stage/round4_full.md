# Review round 4 (full paper, after round-3 fixes)

review_independence: same-family · reviewer: fresh subagent given file paths only · date: 2026-11-23

**Score 5/10 · verdict: almost.** STOP rule (score >= 6) not met; round 4 is the last round allowed by the loop.
acceptance_status remains provisional.

The reviewer recomputed the headline numbers from raw JSON (23%, 7/20, 60-93%, 0.017, 10%, ZTF ratios, SLSN 0.53 -> 0.71)
and found the proofs of Prop. 1, Prop. 2 and Thm. 4 correct as stated.

| # | Severity | Issue | Action |
|---|---|---|---|
| 1 | major | Abstract put GAMUT-cert's guarantee next to default GAMUT's 23% efficiency; validity rule not named | Abstract rewritten: default GAMUT named as empirically validated, bootstrap rule named, validity rates at both levels, cert trade-off stated |
| 2 | major | C1 and temperature scaling have little effect on real data; fixed bins do as well on test folds | Text now states this plainly; Cells column and fixed-bin/in-sample rows added to tab:ablation; merging justified only by validation folds, with the test-fold contradiction stated |
| 3 | major | "controls eq. (1)" overstated | Replaced by cell counts for GAMUT, FLARE, two-step, max-class, MDCP; says the always-valid baselines do better on this target |
| 4 | major | Horizon cells collapse to one at 95% on PLAsTiCC | Realised cells for every benchmark stated in Sec. 3; validity there is per class over all horizons |
| 5 | major | Strict split thin (one origin, no CIs, no MDCP/Kandinsky) | strict_split.py now runs all five ZTF origins with MDCP, Kandinsky, CPTC and the +C1 variants, and the object-level bootstrap of the main text; new summary table tab:strictsum |
| 6 | major | Baselines not given the same components | New baselines MDCP + C1 (GAMUT's residual laws), Two-step CP + C1 and Max-class CP + C1 (class-conditional intervals), on every benchmark |
| 7 | major | Test-set exposure, no held-out set; bootstrap rule lenient | Pre-registered held-out evaluation on PLAsTiCC test chunk 02 (WFD), frozen code, run once (refine-logs/EXPERIMENT_PLAN.md); limitation item on exposure, multiplicity and power |
| 8 | minor | tab:main caption and experiments text overstated cert | Caption and text corrected |
| 9 | minor | Conclusion compared mismatched levels | Both levels given for GAMUT and baselines |
| 10 | minor | "shortest" vs scale-weighted length; Remark evidence weak | Abstract says "smallest scale-weighted length"; remark states the 2-16% absolute-length cost and calls the choice a preference |
| 11 | minor | Empty bands under-reported | Every benchmark with non-negligible empty bands listed in the limitations |
| 12 | minor | Hyperparameter claim selective | Dual-iteration CCG sensitivity reported (macro from validation JSON) |
| 13 | minor | Uniform bound used m_max = 30 | m_max = 267 (measured); slack 2.0 / 0.35 |
| 14 | minor | Thm. 4 conditioning | Theorem and proof condition on the classes and cell memberships of B |
| 15 | minor | Duplicate strict rows in tab:excluded | Fixed in make_tables.py |
| 16 | minor | Sign typo "+3%" | Sentence rewritten |

Also found while fixing: the appendix stated Kandinsky CP's invalid rates as 30%/50%; the main table gives 25%/40%.
Replaced by generated macros.
