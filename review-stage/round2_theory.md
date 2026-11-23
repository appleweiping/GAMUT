# Review round 2: theory and method (after round-1 fixes)

review_independence: same-family · fresh subagent, file paths only · score 5/10 · verdict: almost

| # | Severity | Issue | Response |
|---|---|---|---|
| 1 | critical | Theorem stated per FLARE bin, but the code calibrates merged horizon cells (A-measurable under crc). | Target eq. (1) and the theorem are now per horizon cell (unions of FLARE bins chosen on A). Realised cells are tabulated per setting. |
| 2 | major | Intro and abstract overstate the certificate ("exactly valid ... every class and horizon"). At 95% on the strict split the rare classes are excluded. | Reworded: valid (>= 1-alpha) under a strict split, for classes with enough certification objects. The rare-class exclusion is reported. |
| 3 | major | Label-based validity attributed to the default; "covers every class everywhere". | Validity attributed to GAMUT-cert. The "everywhere" claim is replaced by the bootstrap criterion and its scope. |
| 4 | major | Appendix tables stale (pre-v6); exclusions counted for the wrong variant; the cert caption says "exact". | Tables regenerated after all v6 runs. Exclusions and borrowed cells counted for the default and the cert, cross-conformal and strict. Caption fixed. |
| 5 | major | "Plug-in estimate of the oracle" hides that the classifier posterior is not the Q_b posterior. | The assumption (class-level factors) is stated; "approximates" used elsewhere. |
| 6 | major | The merging rule can collapse horizons. | Listed as a limitation with realised cells; per-bin coverage checked by bootstrap. |
| 7-14 | minor | Prop. 1 idealised examples, Mondrian bracket formula, theorem conditions (A-measurable list, n^B), Prop. 2 density/randomisation/Sadinle analogue, intro omega, text-code mismatches (geometric mean over included classes, merge qualifier), bound (J=10, partition union), grid-length under-count, exchangeability under selection. | All applied in the text; the grid under-count and selection are in the limitations. |
