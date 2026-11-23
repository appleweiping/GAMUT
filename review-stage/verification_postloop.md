# Post-loop verification review (after round-4 fixes, held-out set and closure study)

review_independence: same-family · reviewer: fresh subagent given file paths only · date: 2026-11-23

**Score 6/10 · verdict: almost.** This meets the loop's STOP rule (score >= 6 and verdict in {ready, almost}).
acceptance_status: provisional (same-family review only).

Verified as sound: Prop. 1, Prop. 2 and Thm. 4 and their proofs; the code path of GAMUT-cert matching Thm. 4; all
headline numbers recomputed from raw results (23%, 29%, 100/90% validity, SN Ia exceptions, SLSN 0.53 -> 0.71,
48-66% / 63-80%, 7/20, strict-split, held-out and closure macros); the held-out pre-registration followed (md5 of the
frozen code matches, run once, fully reported); round-4 items 1-6 and 8-16 resolved; 8-page main text.

| # | Severity | Issue | Action |
|---|---|---|---|
| 1 | major | Closure gain credited to the mixture; 25% quoted against the uncalibrated score; C1 trained on more data than FLARE's forecaster | Gain attributed to the class-conditional next-point laws, the mixture's own contribution quantified; headline is the comparison with the same Venn-Abers layer; C1 retrained on FLARE's train split only (`scripts/closure_c1.py`) and the study rerun |
| 2 | major | Registered closure claim only partly met; registered subset not reported; pooled-law rule not registered | Registration-outcome table (tab:closurereg) with every registered comparison and verdict, multi-class subset, deviation disclosed |
| 3 | major | Venn-Abers finite-sample claim unsupported | Claim removed; scores now from train-split models, but T and residual quantiles use val, so no finite-sample claim is made |
| 4 | major | Efficiency endpoint switched from registered class-averaged IS without disclosure | Metrics paragraph states the switch and GAMUT's IS win counts; appendix and design history report the registered endpoint |
| 5 | major | Stale limitation ("no untouched set") | Rewritten to the held-out set's actual scope |
| 6 | minor | Day-5 width ratio | Abstract uses the day-5 ratio |
| 7 | minor | Held-out per-cell result | Reported in the main text |
| 8 | minor | Strict split at 95%: default wider than cert and two-step at day 10 | Stated with the reason (small classes in the minimum) |
| 9 | minor | Iteration-cap disclosure dropped | Restored in Sec. 3 |
| 10 | minor | "methods that also cover every class" | "always-valid baselines" |
| 11 | minor | ElectricDevices zero targets break the density assumption | Limitation added (52-59% exact zeros) |
| 12 | minor | Fixed bins "do as well" understates | "do better at 95%" with numbers |

## Second verification (after the fixes above)

**Score 6/10 · verdict: almost** (same-family). Items 2-12 of the first verification confirmed fixed. Remaining issues
and actions:

| # | Severity | Issue | Action |
|---|---|---|---|
| 1 | major | Closure gain may come from a better next-point law, not class conditioning; rare-class reversal only in appendix | Pooled control run (`scripts/closure_pooled.py`): Brier 0.080/0.064 vs ours 0.077/0.064, so the gain is mainly the learned law; text reframed ("learned next-point law"), SLSN reversal stated in the main text |
| 2 | major | ZTF widths quoted only under cross-fitting | Strict-split widths (1.08/1.70) added in the introduction |
| 3 | major | Registered plan outcomes reported only in part; B6 not run; B1 paired IS test missing | Registered-outcome table for C1-C5 and B0-B8 (tab:plan); B1 paired IS bootstrap run (`scripts/is_bootstrap.py`): better in 3/20 and 5/20, worse in 13/20 and 14/20; B6 reported as not run |
| 4 | minor | "always-valid baselines do better" on cells is false for MDCP | Two-step and max-class do better, MDCP worse (1.8/1.9) |
| 5 | minor | Prop. 1 overstated in abstract | "plus the posterior's own error" |
| 6 | minor | Certificate scope | "(rare classes may have too few)" |
| 7 | minor | 23% average hides weak cases | Per-level gains (27%/20%) in the abstract; MDCP-at-95% 8% in README and tab:plan |
| 8 | minor | Wording (CV at day 20; which hyperparameters) | Fixed |
| 9 | minor | Closure per-class ECE and subset intervals missing | Added (tab:closureclass, tab:closurereg) |
| 10 | minor | Stale numbers in README and design log | Updated |
| 11 | minor | No limitation in the conclusion | Added |
