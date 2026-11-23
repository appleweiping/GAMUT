# Lit sweep: conformal regression / multi-horizon forecasting as external baselines (2026-11-22)

Angle: post-hoc baselines on shared FLARE-style outputs (mu, q16/q84, q025/q975, class probs p(x)), per horizon.
Verification: every entry was checked this session against the arXiv export API (title/authors/abstract) and
a venue page (PMLR / NeurIPS / ICLR proceedings pages, JMLR page, or Crossref DOI record). "unverified" = venue not confirmed.

| # | Paper | Venue (how checked) | Role |
|---|-------|---------------------|------|
| 1 | Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. Distribution-Free Predictive Inference for Regression | JASA 2018 (Crossref 10.1080/01621459.2017.1307116) | baseline: split + locally weighted CP |
| 2 | Romano, Patterson, Candes. Conformalized Quantile Regression | NeurIPS 2019 (proceedings metadata json) | baseline |
| 3 | Chernozhukov, Wuthrich, Zhu. Distributional conformal prediction | PNAS 2021 (Crossref 10.1073/pnas.2107794118) | optional baseline |
| 4 | Stankeviciute, Alaa, van der Schaar. Conformal Time-series Forecasting (CF-RNN) | NeurIPS 2021 (proceedings page) | joint-horizon baseline |
| 5 | Xu, Xie. Conformal prediction interval for dynamic time-series (EnbPI) | ICML 2021 PMLR 139 (PMLR page); TPAMI 2023 journal (Crossref) | related work |
| 6 | Xu, Xie. Sequential Predictive Conformal Inference for Time Series (SPCI) | ICML 2023 PMLR 202 (PMLR page) | related work / optional |
| 7 | Gibbs, Candes. Adaptive Conformal Inference Under Distribution Shift (ACI) | NeurIPS 2021 (proceedings page) | related / optional online |
| 8 | Angelopoulos, Candes, Tibshirani. Conformal PID Control for Time Series Prediction | NeurIPS 2023 (Crossref 10.52202/075280-1000) | related / optional online |
| 9 | Guan. Localized conformal prediction: a generalized inference framework | Biometrika 110(1):33-50 (Crossref 10.1093/biomet/asac040) | baseline variant |
| 10 | Hore, Barber. Conformal prediction with local weights (RLCP) | JRSSB 2024 (Crossref 10.1093/jrsssb/qkae103; journal title says "robust guarantees") | baseline |
| 11 | Bostrom, Johansson. Mondrian conformal regressors | COPA 2020, PMLR 128 (PMLR page); follow-up overlapping categories PMLR 329 (2026) | incumbent baseline (FLARE) |
| 12 | Podkopaev, Ramdas. Distribution-free UQ for classification under label shift | UAI 2021, PMLR 161 (PMLR page) | shift baseline (PLAsTiCC) |
| 13 | Tibshirani, Barber, Candes, Ramdas. Conformal Prediction Under Covariate Shift | NeurIPS 2019 (proceedings page) | theory tool |
| 14 | Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control | ICLR 2024 (proceedings page) | theory tool for C3 |
| 15 | Dunn, Wasserman, Ramdas. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models | JASA (Crossref 10.1080/01621459.2022.2060112, online 2022) | protocol / theory |
| 16 | Lee, Barber, Willett. Distribution-free inference with hierarchical data | ACM J. Data Science 2026 (Crossref 10.1145/3786352) | protocol / theory |
| 17 | Gibbs, Cherian, Candes. Conformal Prediction With Conditional Guarantees | JRSSB 2025 (Crossref 10.1093/jrsssb/qkaf008) | key baseline (GCC with p(x)) |
| 18 | Bairaktari, Wu, Wu. Kandinsky Conformal Prediction | ICML 2025, PMLR 267 (PMLR page) | CLOSEST COMPETITOR |
| 19 | Zhang, Candes. Posterior Conformal Prediction | arXiv 2409.19712 (v2 2026), venue unverified | key baseline |
| 20 | Sun, Yu. Conformal Prediction for Time-series Forecasting with Change Points (CPTC) | NeurIPS 2025 (Crossref 10.52202/085713-0410) | CLOSE COMPETITOR (state-conditional forecaster + p-weighted union) |
| 21 | Zhou, Lindemann, Sesia. Conformalized Adaptive Forecasting of Heterogeneous Trajectories (CAFHT) | ICML 2024, PMLR 235 (PMLR page) | joint-band baseline |
| 22 | Plassier et al. Rectifying Conformity Scores for Better Conditional Coverage (RCP) | ICML 2025, PMLR 267 (PMLR page) | baseline |
| 23 | Ding, Angelopoulos, Bates, Jordan, Tibshirani. Class-Conditional CP with Many Classes | NeurIPS 2023 (Crossref) | tool for rare classes |
| 24 | Izbicki, Shimizu, Stern. CD-split and HPD-split | JMLR 23 (2022) (JMLR page) | baseline (union-of-intervals) |
| 25 | Wang, Hyndman. Online conformal inference for multi-step time series forecasting (AcMCP) | arXiv 2410.13115, venue unverified | related work (per-horizon online) |

## Facts that matter for PRISM (seen in full text this session)
- Kandinsky (arXiv 2502.17264v2, Sec. 3.1 "Fractional Group Membership", Cor. 3.4): with Phi_G(x,y)=P[Z in G | phi(X,Y)],
  phi=(X,Y) or (f(X),Y) sufficient for the score, the set {y: S(x,y) <= q(x,y)} satisfies, w.p. >= 1-delta,
  |P_G^Z(cover) - (1-alpha)| <= (1/pi_G^Z)(C sqrt(|G|/n) + |G|/n + max{alpha,1-alpha} sqrt(2 ln(4|G|/delta)/n)).
  => true-group coverage, but asymptotic-rate (not exact), assumes Phi known, 1/pi_G blow-up for rare classes,
  test-time search over continuous y (A.3: "complex ... for continuous label domains").
  PRISM must compare against this and state the difference (exact finite-sample per-class, no P(Z|X,Y) model).
- GCC / Kandinsky with Phi(x)=p(x) only: weighted coverage deviation wCD = E[w(X,Y)(1{Y in C} - (1-alpha))] -> 0,
  i.e. the probability-weighted guarantee PRISM's key observation refers to.
- PCP Theorem 1: P{Y not in C(X_{n+1}) | pi*(X_{n+1})} <= alpha, where pi* is an m-draw multinomial randomization of the
  membership vector pi(X). Conditional on randomized membership, not on the latent cluster.
- CPTC Alg. 1: forecaster f: X x Z -> Y; per-state score sets S_z with ACI-updated alpha_{z,t}; aggregation
  Gamma_t = {y: sum_z p(z|x) 1{y in Gamma_z} >= 1-alpha} (Eq. 10). Prop 4.1: marginal validity under exchangeability;
  Thm 4.2: asymptotic long-run validity. Same C1 idea as PRISM; guarantee is marginal, not per true state.
