# Review round 1: theory and method

review_independence: same-family · reviewer: fresh subagent, given file paths only · score 4/10 · verdict: not ready

The reviewer found the core algebra correct: the Prop. gap identity, the Lagrangian duality, CRC with the min over
classes, and the Sauer-Shelah/Massart arithmetic. The problems are in how the results are applied and what is
claimed. Issues (severity, summary) and the response to each:

| # | Severity | Issue | Response |
|---|---|---|---|
| 1 | critical | Thm (a)/(b) assume p-hat, f-hat independent of the calibration objects. Default GAMUT fits the temperature, the residual laws (LOO-scored) and v in-sample, so (b)'s fixed half-space class does not apply. | Thm restated for GAMUT-cert (split) only. Default GAMUT is presented as a plug-in estimate of the oracle with no finite-sample claim, validated empirically. LOO is described as a bias correction, not a guarantee. |
| 2 | critical | Cross-fitted base models: calibration objects' outputs come from models trained on the test fold, so even GAMUT-cert is outside (a) in the cross-conformal protocol. | The paper states that the main protocol (FLARE's) is cross-conformal for every method. Added a strict three-way-split experiment (train / calibrate / test disjoint) on ZTF to test the exactness claim. |
| 3 | major | A per-bin guarantee does not imply eq. (1) (pooled over bins); counterexample, and SN_CC object coverage is 0.668 overall while every bin is >= 0.678. | The target is now defined per (class, lead-time bin), the professor's "per horizon" requirement. Pooled per-class coverage is reported as FLARE's metric, without a guarantee claim. |
| 4 | major | The headline "valid" metric (pooled per-class) is not the guaranteed quantity. | Validity is judged by object-level bootstrap per (class, bin) cell and per class. Both are reported. |
| 5 | major | Exclusion cutoff off by one (needs n >= 1/alpha - 1). Excluded cells not listed. Nearest-bin fallback undisclosed. | Fixed in code (v4). Excluded cells and fallback bins are recorded per fold and tabulated. |
| 6 | major | Prop. gap applications overreach: GCC is exact only for the true p_c in the span and in the augmented randomised form; p-hat needs a correction term; mixtures unproven; Mondrian gives a bracket; the sign is wrong in the intro; the abstract claim is too broad. | All corrected; correction term added; mixture claim dropped; bracket stated. |
| 7 | major | The dual update is a heuristic, not the oracle's dual ascent. The KKT claim is informal. Slack classes run the loop to the cap. | Called a heuristic. Cap and measured iteration counts reported (from recorded diagnostics). KKT claim rephrased. |
| 8 | major | Sampling measure unspecified; the classifier posterior is not the point-level posterior; consistency sketch is unsupported. | Measure defined. Posterior described as a proxy whose error costs width, not validity. Consistency claim removed. |
| 9 | major | Remark "absolute length" contradicts the per-bin formulation; the 0.39 is unsupported. | Rewritten as a within-bin statement, with numbers from the ablation table. |
| 10 | major | Thm (b) is numerically vacuous at these class sizes. | (b) moved to the appendix with its numerical slack stated; the main text no longer relies on it. |
| 11-22 | minor | Proof constants, K=1 claim false, readings of the oracle, bias absorption, "no training beyond base forecaster" false, runtime and iterations unsupported, proof gaps (existence, rho_g > 0), coverage vs length grid mismatch, LOO wording, Algorithm 1 omissions, citation numbers. | Each corrected in the text. Runtime and iterations come from recorded measurements. Grid pruning is described. |
