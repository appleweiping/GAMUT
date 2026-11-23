# Lit sweep: coverage conditional on a latent / test-time-unknown group (2019-2026)

Sweep date: 2026-11-22. Angle: coverage conditional on a class/group Z that is known in calibration
but unobserved at test time, for regression or forecasting targets, compared with PRISM
(C1 hypothesis-conditional forecaster, C2 true-class Mondrian class set Gamma_beta, C3 union over Gamma
with per-true-class CRC, lambda* = max_c).

How entries were checked: every entry below was seen this session through the export.arxiv.org API
(title, authors, abstract), arXiv/ar5iv HTML full text (for the quoted theorems), Crossref, or web-search
result pages (proceedings / mlanthology / PMLR / NeurIPS / ICLR listings). WebFetch could not resolve
arxiv.org, export.arxiv.org or proceedings.mlr.press. curl through Bash worked for export.arxiv.org,
arxiv.org/html, ar5iv and Crossref. OpenReview api2 returned a challenge (403). dblp returned 202.
"verified" means venue and year were confirmed on a page.

## Most important finding (read first)

**Kandinsky CP (Bairaktari, Wu, Wu, ICML 2025) is the closest competitor and needs a direct response.**
Its "Fractional Group Membership" application targets coverage conditional on *unobserved* attributes Z.
It uses basis functions Phi_G(x,y) = P[Z in G | phi(X,Y) = phi(x,y)], where phi is (X,Y) or (f(X),Y).
Its ACSIncome *regression* experiment (CQR score, state omitted at test time) evaluates exactly the
per-latent-group miscoverage. Guarantee: |coverage given Z in G - (1-alpha)| = O(sqrt(|G|/n) + |G|/n)
w.h.p., assuming Phi is known and phi is a sufficient statistic for the score.

Consequence for PRISM's "key observation": the tower property gives
E[1{Z=c} * miss] = E[P(Z=c | X,Y) * miss], because miss is a function of (X, Y, eps).
So weighting by a posterior that **includes the target Y** recovers true-class coverage exactly.
The gap E[Cov(1{Z=c}, miss | X)] is specific to weights that depend on X only (GCC with p(x), BatchGCP,
Posterior CP, MoE-CP). The paper must say this precisely. Claimed advantages over Kandinsky that still
hold, and that must be shown empirically:
- PRISM's guarantee is exact and finite-sample (>= 1-alpha per class) with no estimated posterior
  P(Z | prefix, future).
- PRISM needs no y-grid. Kandinsky builds the set as {y : S(x,y) <= q_hat(x,y)}, and its limitations
  note that regression needs a search over y.
- PRISM gives one joint band across horizons.
- Kandinsky's rate degrades with the number of groups and with error in Phi.

Kandinsky-FY/XY with a GBT posterior on (prefix-forecast, future) must be a baseline.

## Trivially valid "union / max" baselines that PRISM must beat on width
- Barber, Candes, Ramdas, Tibshirani (2020/21): the oracle uses "the maximum quantile over any set
  containing the point x". With a latent Z, every class "contains" x, so this becomes the max over all
  classes of the per-true-class quantile. Kandinsky's experiments run it as "conservative CP".
- The union over **all** c of class-c Mondrian (true-label) bands is valid for every class, because
  union contains C_c. This is PRISM with Gamma = all classes, so it doubles as the screening ablation.
- PRISM's novelty over these is (i) screening to Gamma_beta(x), (ii) class-specific centres mu_c and
  scales sigma_c, and (iii) one joint lambda that keeps exact 1-alpha rather than 1-alpha-beta.
  If C3 calibrates the final union directly, the beta of screening is absorbed. Verify this in the proof.

## Paper notes (abridged; full structured list returned to orchestrator)
1. Gibbs, Cherian, Candes. Conformal prediction with conditional guarantees. JRSSB 87:1100-1126, 2025
   (doi 10.1093/jrsssb/qkaf008; arXiv 2305.12616).
   Thm 2: for F = {Phi(.)^T beta}, P_f(Y in C(X)) >= 1-alpha for every nonnegative f in F.
   Cor 1 (groups): P(Y in C | X in G) >= 1-alpha, with upper bound 1-alpha + |G|/((n+1)P(G)).
   With Phi = p_hat(x) this gives only p-tilted coverage. Baseline "GCC-p".
2. Bairaktari, Wu, Wu. Kandinsky CP. ICML 2025, PMLR 267:2581-2602 (arXiv 2502.17264). See above.
3. Romano, Barber, Sabatti, Candes. Equalized coverage. HDSR 2020 (doi 10.1162/99608f92.03f00592).
   P(Y in C(X,A) | A=a) >= 1-alpha. **Requires A_{n+1} at test time** ("test point with known X_{n+1}
   and A_{n+1}"). Groupwise QR training is the analogue of C1. Oracle upper bound.
4. Zhou, Sesia. AFCP. NeurIPS 2024 (arXiv 2405.15106). Classification with an observed attribute.
   Selection uses placeholder labels y, and the final set is a union over y of group-conditional sets,
   the same validity-preserving "union over hypotheses" device.
5. Barber, Candes, Ramdas, Tibshirani. The limits of distribution-free conditional predictive inference.
   Information and Inference 2020 online (doi 10.1093/imaiai/iaaa017). Impossibility result plus the
   max-quantile construction above.
6. Jung, Noarov, Ramalingam, Roth. Batch multivalid CP (BatchGCP/BatchMVP). ICLR 2023 (arXiv 2209.15145).
   Groups are regions of feature space only.
7. Wang, Yaldiz, Bakman, Karimireddy. CP adaptive to unknown subpopulation shifts. arXiv 2506.05583
   (2025; submitted to ICLR 2026). Thm 2.1: with a domain classifier of per-domain accuracy gamma,
   P(Y in C | X ~ P_k) <= max(0, gamma - alpha) is possible. This formalises FLARE's argmax-Mondrian
   failure. Their fix (classifier-probability-weighted CP) is valid for Bayes-optimal or
   multicalibrated classifiers and targets coverage per test *environment*, not per latent class.
8. Zhang, Candes. Posterior conformal prediction. arXiv 2409.19712 (2024, v2 2026). Scores are modelled
   as a mixture R | X ~ sum_k pi_k(X) f_k, and coverage holds conditional on pi(X) (eq. 7), not on
   the latent cluster. Baseline. Code: github.com/yaozhang24/pcp
9. Stutz et al. CP under ambiguous ground truth. TMLR 2023. Monte Carlo CP gives a guarantee with
   respect to P_agg. Classification only.
10. Sesia, Wang, Tong. Adaptive conformal classification with noisy labels. JRSSB 2024
    (doi 10.1093/jrsssb/qkae114). Classification only. Relevant to noisy spectroscopic labels in C2/C3.
11. Guan, Tibshirani. BCOPS. JRSSB 2022 (doi 10.1111/rssb.12443). Per-class sets that may be empty,
    for outliers. A tool for C2 with abstention.
12. Podkopaev, Ramdas. UQ under label shift. UAI 2021, PMLR 161:844-853. Under label shift, true-class
    conditional validity is invariant to class priors. Use this in the PLAsTiCC train-to-test argument.
13. Ding, Angelopoulos, Bates, Jordan, Tibshirani. Class-conditional CP with many classes. NeurIPS 2023.
    Clustered CP pools rare classes, which matters for lambda* = max_c when rare classes have tiny n_c.
14. Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal risk control. ICLR 2024. E[L(lambda_hat)] <= alpha
    for monotone bounded losses, tight to O(1/n). Per-class CRC with max_c is the core of C3.
15. Javanmard, Shao, Bien. Prediction sets for high-dimensional mixture of experts. JRSSB 87(3):850-871
    (per the NSF PAR record; year not shown on the page seen; arXiv 2210.16710). The set is a union of
    K intervals from the gating-weighted mixture density. Validity is asymptotic and parametric, and
    marginal over the latent expert. Closest *geometric* precedent to the PRISM union.
16. Kong et al. MoE-CP (gating-similarity weighted CP). ICLR 2026. Soft assignments only. Marginal
    validity is kept. The output is a single interval.
17. Izbicki, Shimizu, Stern. CD-split / HPD-split. JMLR 23(87), 2022. Unions of intervals with local and
    asymptotic conditional validity. Use it as the "probability-weighted mixture of class-wise bands"
    baseline: HPD-split on sum_c p_c(x) N(mu_c, sigma_c).
18. Wang, Gao, Yin, Zhou, Blei. Probabilistic CP (union of balls around generative samples).
    AISTATS 2023, PMLR 206. Finite-sample marginal validity only.
19. Gasparin, Ramdas. Merging uncertainty sets via majority vote. arXiv 2401.09379 (2024). Merging K
    arbitrarily dependent 1-alpha sets gives a set with >= 1-2alpha coverage. Weighted vote variant.
20. Wong, Prinster, Saria, Chellappa, Liu. Weighted aggregation of prediction sets with data-dependent
    weights. AISTATS 2026 (arXiv 2505.11785). Coverage between 1-2alpha and 1-alpha. MoE experiments.
21. Bhattacharyya, Ding, Barber. CP with macro-coverage guarantees. arXiv 2606.28598 (2026).
    Thm 1: MacroCov >= 1 - alpha - max_k w(k)/N_k. Gives an alternative "average over true classes"
    target.
22. Bhattacharyya, Barber. Group-weighted CP. arXiv 2401.17452 (2024). The shift is defined at the
    level of finite groups.
23. Sadinle, Lei, Wasserman. Least ambiguous set-valued classifiers. JASA 114(525):223-234, 2019.
    Per-class thresholds on p_c(x) give Gamma_beta (C2).
24. Combes, Coeurjolly, Dervilly. Macro Mondrian CRC. arXiv 2610.35035 (2026). Group-reweighted CRC
    for rare groups. A macro-average, not a max.
25. Wang, Cheng, Guo, Liu, Yu. Equal opportunity of coverage in fair regression (BFQR). NeurIPS 2023
    (per arXiv comment). Regression, with groups conditioned on the true outcome.

Extras seen, not in the main list:
- Bastani et al., MVP, NeurIPS 2022 (adversarial multivalid).
- Deng, Ardeshir, Hsu 2023 (arXiv 2303.03995; hidden stratification via multi-group learning).
- Sasli et al. ASTRANet 2026 (arXiv 2607.08044; AD-stratified Mondrian CP for spectroscopic
  transients).
- Walls et al. RASTI 2026 (Monte Carlo CP for radio galaxies under ambiguous ground truth).
- Dewolf, De Baets, Waegeman 2023 (arXiv 2309.08313; Mondrian taxonomy kappa(x,y) for regression).
- Braun et al. 2025 (arXiv 2512.11779; ERT conditional-coverage diagnostics).
- Min, Peng, Zou 2026 (arXiv 2605.11602; decomposition of conditional miscoverage).
- Laplante 2026 (arXiv 2605.25852; PIT-CP).

## Gaps (for the contract)
- No paper found gives an *exact finite-sample* guarantee P(Y in C(X) | Z=c) >= 1-alpha for every
  latent class in regression or forecasting without estimating P(Z | X, Y). Kandinsky is approximate
  and needs that posterior. Wang et al. 2025 target environments, not classes. PCP, GCC-p and MoE-CP
  condition on probabilities only.
- None of the latent-group papers treats multi-horizon or simultaneous forecast bands, or prefix-length
  dependence.
- With rare classes (n_c of tens), lambda* = max_c is driven by the rarest class. No paper found
  analyses the width cost of worst-class CRC versus clustering (Ding) or macro targets (Bhattacharyya
  et al.).
- Noisy or ambiguous class labels in calibration are handled only for classification (Sesia, Stutz).
- "Fair CP with unknown sensitive attribute" for regression is covered only by Kandinsky's fractional
  membership. No dedicated method was found.
