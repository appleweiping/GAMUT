# Lit sweep: astronomical light-curve forecasting with uncertainty, and benchmarks (2018-2026)

Sweep date: 2026-11-22. Angle: light-curve extrapolation and next-observation prediction for
ZTF/LSST transients; calibrated or conformal bands in astronomy; follow-up scheduling that uses
forecasts; PLAsTiCC / ELAsTiCC data and class lists; UCR/UEA datasets used for forecasting or
early classification; whether the name "PRISM" is free.

How entries were checked: each paper was seen this session through the export.arxiv.org API
(title, authors, abstract, journal-ref, DOI). Venues were checked on Crossref DOI records, or on
the paper PDF footer (NeurIPS 2019, TMLR 11/2025), or on the neurips.cc virtual page (NightLANP).
Full-text quotes come from arXiv PDFs read with pdftotext (Muthukrishna 2022, NightLANP, NAPTIME,
DILATE, soft-DTW, Renault 2025). The PLAsTiCC class table comes from the Zenodo release notes
(note1_dataRelease.pdf and note2_modelNames.pdf, record 2539456). The ELAsTiCC details come from
the NERSC portal page and its ELAsTiCC2 A_MODEL_SUMMARY.TXT. The UCR dataset facts come from
timeseriesclassification.com description pages.
WebFetch could not resolve timeseriesclassification.com, icml.cc or neurips.cc. curl through Bash
worked for all three. The GitHub API for LSSTDESC/elasticc is not enabled in this session, so the
taxonomy notebook was not read.
"verified" means the venue and year were confirmed on a page. Entries marked unverified are arXiv-only.

## Key findings for PRISM (read first)

1. **Per-class forecasters already exist, but not with coverage.** Muthukrishna et al. (MNRAS 2022)
   "trained five autoregressive models one for each transient class: SNIa, SNII, SNIbc, SLSN, TDE.
   And similarly, for the Bazin function, we defined a prior distribution for each class". Each
   model forecasts P(F(T+3) | D(t<=T)). The per-class scores are used for anomaly detection and are
   combined as a **mean** over the "common" classes. No band is calibrated and no coverage is
   claimed. This is the nearest astronomy precedent for C1 (querying a forecaster under every class
   hypothesis). PRISM should cite it and say what is new: the union over a conformal class set
   (C2) and the per-true-class risk control (C3).
2. **No astronomy paper found gives forecast bands with coverage conditional on the true class.**
   - NightLANP (NeurIPS 2026 poster) reports PICP@95% and PIT pooled over 15 classes. Its training
     and test sets are class-balanced (1,000 and 250 light curves per class), so its evaluation cannot
     show rare-class under-coverage under the natural class imbalance. The authors also state that
     the model "(like other models) it was struggling at forecasting tasks from rise-only
     observations", and they train a separate forecasting version.
   - FLARE calibrates per predicted class and under-covers SLSNe and late CVs.
   - REFITT scores a forecast by "prediction confidence" but gives no coverage.
3. **Conformal work in astronomy is mostly classification sets or galaxy regression.** Examples:
   Cui+ 2024 (ApJS) light-curve classification sets; ASTRANet (2026) "AD-stratified Mondrian";
   the AION-1 UQ benchmark (CQR, LVD). Apart from FLARE, none applies conformal prediction to
   light-curve forecasts.
4. **Prefix protocols are standard in transient early classification:**
   - SCONE truncates light curves at 0/5/15/25/50 days after trigger.
   - RAPID classifies "from within a day of the initial alert".
   - First Impressions reports accuracy within 3 days of discovery.
   - ORACLE classifies from 1 day after first detection.
   - NAPTIME uses prefix sweeps ("earliest 10%").
   No paper found defines a prefix-to-future forecasting benchmark with a latent class on PLAsTiCC
   or ELAsTiCC. PRISM would have to define one and state the protocol exactly.
5. **UCR prefix-to-future forecasting has precedent, but with leakage and class caveats.**
   - DILATE (NeurIPS 2019) uses ECG5000: "We take the first 84 time steps (60 %) as input and
     predict the last 56 steps (40 %)". It reuses the soft-DTW setup (ICML 2017): "the first 60% of
     the time series as input and the remaining 40% as output, ignoring class information".
   - Renault et al. (TMLR 2025) benchmark early classification on prefixes of 77 UCR datasets,
     including ECG5000, ElectricDevices, StarLightCurves and Wafer. They warn that "z-normalization
     would require knowledge of the entire incoming time series, but it could also introduce a
     leakage of information from the future of the time series to the previous time steps".
     -> PRISM must re-normalise each prefix itself, or use raw series, or report this as a limitation.
   - StarLightCurves is "(purportedly) phase-aligned", so the "future" there is a later phase of a
     folded period, not later time. The paper must describe it as phase-segment forecasting.
   - ECG5000 labels "were obtained by automated annotation". Its true class is therefore a
     label-noise proxy.
6. **The name PRISM is crowded. Rename.** The collisions are in astronomy and in time series:
   - van der Velden+ 2019 (ApJS 242, 22) "PRISM, a 'Probabilistic Regression Instrument for
     Simulating Models'". This is an astronomy uncertainty-analysis emulator.
   - JWST NIRSpec/PRISM time-variability light curves (e.g., arXiv 2509.13544).
   - Roman "prism time series" for SN cosmology (arXiv 2306.17228, 2206.10632).
   - Time-series methods named PRISM: forecasting (arXiv 2512.24898, 2603.25378, SEPNET-PRISM-R
     2610.28292); classification (2508.04503); anomaly detection (2608.03926).
   - A reliability and abstention framework for sequential prediction (arXiv 2611.18127) and an
     LLM-judge calibration method (2610.18389).
   No conformal method titled PRISM was found (arXiv title+abstract search, web search).
   Proposed alternatives. None of these was found in arXiv titles in stat.ML, cs.LG, stat.ME or
   astro-ph.IM on 2026-11-22. This is not a trademark check.
   - **GAMUT** (Guaranteed All-class Mondrian Union Tubes). It keeps the spectrum metaphor. Arxiv
     titles using the word are about colour gamut and one agent benchmark ("ANI-Gamut").
   - **TRUBAND** (TRUe-class conformal BANDs). No title hits.
   - **PANOPLY** (Plausible-hypothesis ANd Oracle-class Prediction bands). The only title hit is the
     common-noun use "a panoply of quantum objects".
   - Rejected: TRUCE (used by an LLM private-benchmarking paper, arXiv 2403.00393), QUORUM (used by
     a 2025 quantum anomaly-detection method), CANDOR and CHORUS (both used for ML methods in 2024-26).

## Papers (25)

| # | Paper | Venue (how checked) | Role |
|---|---|---|---|
| 1 | Muthukrishna, Mandel, Lochner, Webb, Narayan. Real-time detection of anomalies in large-scale transient surveys | MNRAS 517, 393 (2022), Crossref 10.1093/mnras/stac2582 | related work + baseline (per-class Bazin/TCN predictive) |
| 2 | Sravan, Milisavljevic, Reynolds, Lentner, Linvill. Real-time, Value-driven Data Augmentation in the Era of LSST (REFITT) | ApJ 893, 127 (2020), Crossref | related work (forecast-driven follow-up) |
| 3 | Chaini, Bianco, Mahabal. Neural Process Family ... NightLANP | NeurIPS 2026 poster (neurips.cc page); ICML 2026 AI4Physics workshop | competitor / external baseline (ANP; the group already tried an NP as a method) |
| 4 | Earl, Chaini, French, Hinkle, ... NAPTIME | ApJ 1010, 11 (2026), Crossref | related work |
| 5 | Boone. ParSNIP | AJ 162, 275 (2021), Crossref | baseline (generative forecaster) |
| 6 | Boone. Avocado (GP augmentation) | AJ 158, 257 (2019), Crossref | baseline (2D GP extrapolation) |
| 7 | Villar et al. Supernova Photometric Classification Pipelines ... PS1-MDS | ApJ 884, 83 (2019), Crossref | baseline (Villar parametric fit) |
| 8 | de Soto, Villar, Berger, Gomez, ... Superphot+ | ApJ 974, 169 (2024), Crossref | baseline (ZTF parametric fits) |
| 9 | Demianenko et al. NN approaches for transient light curve approximation | A&A 677, A16 (2023), Crossref | baseline / related (MLP, BNN, NF on ZTF BTS and PLAsTiCC) |
| 10 | Qu, Sako. Early-time SN light curves with SCONE | AJ 163, 57 (2022), Crossref | benchmark protocol (prefix truncation) |
| 11 | Gagliano et al. First Impressions | ApJ 954, 6 (2023), Crossref | related work (early ZTF BTS) |
| 12 | Muthukrishna et al. RAPID | PASP 131, 118002 (2019), Crossref | related / early-class baseline |
| 13 | Junell, Sasli, Nunes, Xu, ... AppleCiDEr I | PASP 138, 054508 (2026), Crossref | context (parent system of FLARE) |
| 14 | Sasli et al. FLARE | arXiv 2610.30656 (unverified venue) | incumbent baseline |
| 15 | Gomez et al. FLEET | ApJ 904, 74 (2020), Crossref | related work (early SLSN follow-up) |
| 16 | Cui, Armstrong, Feng. Light-curve classification framework II | ApJS 274, 29 (2024), Crossref | related work (conformal sets on light curves) |
| 17 | Sasli et al. ASTRANet | arXiv 2607.08044 (unverified) | related work (AD-stratified Mondrian CP) |
| 18 | Tame-Narvaez, Ciprijanovic, Trivedi. UQ on AION-1 | arXiv 2606.07771, PAI 2026 contribution (unverified) | related work (CQR/LVD in astronomy) |
| 19 | Kessler et al. PLAsTiCC models and simulations | PASP 131, 094501 (2019), Crossref | benchmark |
| 20 | Hlozek et al. PLAsTiCC results | ApJS 267, 25 (2023), Crossref | benchmark |
| 21 | Shah, Gagliano, Malanchev, Narayan, ... ORACLE | ApJ 995, 4 (2025), Crossref | benchmark use of ELAsTiCC |
| 22 | Perley et al. ZTF BTS II | ApJ 904, 35 (2020), Crossref | benchmark (BTS sample) |
| 23 | Dau et al. The UCR time series archive | IEEE/CAA JAS 6(6) 1293 (2019), Crossref | benchmark |
| 24 | Renault, Bondu, Cornuejols, Lemaire. Early Classification of Time Series: Survey and Benchmark | TMLR 2025 (arXiv journal-ref + PDF footer 11/2025) | benchmark protocol |
| 25 | Le Guen, Thome. DILATE | NeurIPS 2019 (PDF footer) | benchmark protocol (ECG5000 84->56) |

Seen this session but not counted above:
- Sravan+ 2022, LNCS, pp. 59-72: autonomous follow-up by value iteration.
- Plestkova+ 2025, arXiv 2507.11785: BiLSTM kilonova light-curve forecasts for ZTF and Rubin.
- Gondhalekar, Moller, Sanchez-Saez 2026, arXiv 2605.24095: mTAN interpolation of Fink/ZTF light
  curves, about 0.01 mag bias and 0.1 mag scatter.
- Cvorovic-Hajdinjak+ 2022, Astron. Nachr. 343: conditional neural process for AGN light curves.
- Fraga+ 2024, A&A 692, A208: Fink classifiers on ELAsTiCC.
- Zhou+ 2026, ApJ 1006, 223: ELAsTiCC classification histories.
- Ishida+ 2019, MNRAS 483, 2: active-learning spectroscopic follow-up.
- Fremling+ 2020, ApJ 895, 32: BTS I.
- Gupta+ 2020, IEEE TAI: early-classification review.
- Schafer & Leser 2020, DMKD 34, 1336: TEASER.
- Bagnall+ 2018, arXiv 1811.00075: UEA archive.
- Cuturi & Blondel 2017, ICML: soft-DTW, with UCR 60/40 multistep forecasting.
- Giertych+ 2024, arXiv 2412.10544: conformal prediction with measurement error.
- Walls+ 2026, arXiv 2603.20000: MC conformal prediction under ambiguous labels.
- Dey+ (LADaR / Cal-PIT), arXiv 2205.14568: local recalibration for photo-z; unverified venue.
- Bazin+ 2009, A&A 499, 653: the Bazin function. It predates 2018 but is the standard parametric baseline.
- van der Velden+ 2019, ApJS 242, 22: PRISM emulator (name collision).

## Data facts checked this session

**PLAsTiCC** (Zenodo 10.5281/zenodo.2539456, "Unblinded Data for PLAsTiCC Classification Challenge", 2019-01-21)
- Files:
  - plasticc_train_lightcurves.csv.gz (21.6 MB)
  - plasticc_train_metadata.csv.gz
  - plasticc_test_metadata.csv.gz (157 MB)
  - plasticc_test_lightcurves_01..11.csv.gz. File 01 is "test set, DDF only" (148.5 MB); files 02-11
    are WFD (about 711 MB each).
- The metadata has `ddf_bool`, `true_target`, `true_submodel`, `true_z` and `true_peakmjd`.
- Light-curve columns: object_id, mjd, passband 0-5 = ugrizy, flux (zeropoint 27.5), flux_err, detected_bool.
- Class table (num: name, N train, N test):
  - Supernovae: 90 SNIa 2,313 / 1,659,831; 67 SNIa-91bg 208 / 40,193; 52 SNIax 183 / 63,664;
    42 SNII 1,193 / 1,000,150; 62 SNIbc 484 / 175,094; 95 SLSN-I 175 / 35,782.
  - Other extragalactic transients: 15 TDE 495 / 13,555; 64 KN 100 / 131; 88 AGN 370 / 101,424.
  - Galactic variables: 92 RRL 239 / 197,155; 65 M-dwarf 981 / 93,494; 16 EB 924 / 96,572;
    53 Mira 30 / 1,453; 6 muLens-Single 151 / 1,303.
  - Test only (class 99 during the challenge): 991 muLens-Binary 0 / 533; 992 ILOT 0 / 1,702;
    993 CaRT 0 / 9,680; 994 PISN 0 / 1,172; 995 muLens-String 0 / 0.
  - TOTAL 7,846 train / 3,492,888 test.
- Consequences for PRISM:
  1. Training and test class proportions differ strongly; the training set is "non-representative".
     Calibrating on training and testing on DDF is label shift as well as covariate shift.
  2. The 99-classes have no training labels. A latent-class guarantee can only cover the 14 trained
     classes. Class-99 objects must be excluded or reported separately as out-of-taxonomy.
  3. Kessler+ 2019: "18 transient and variable source models"; 1,094 teams.

**ELAsTiCC / ELAsTiCC2** (portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/)
- About 4 million simulated objects. ELAsTiCC2 has about 50 million detections and about 400 million
  forced-photometry points. It uses LSST baseline 3.2 cadence with rolling cadence and DDF fields.
- Training samples are public over HTTPS:
  - ELAsTiCC: FULL_ELASTICC_TRAIN.tar, 7.3 GiB.
  - ELAsTiCC2: ELASTICC2_TRAINING_SAMPLE_2/ELASTICC2_TRAIN_02.tar.bz2, 7.4 GiB, SNANA FITS
    HEAD/PHOT files by model, with truth in the *.DUMP files.
- Full streamed data and truth tables are at NERSC or in the DESC TOM, which requires an account.
  The ELAsTiCC truth CSVs are linked on the page.
- The ELAsTiCC2 training sample lists 32 SNANA models and writes 1,474,844 light curves:
  - KN_K17, KN_B19
  - SLSN-I+host, SLSN-I_no_host, PISN, ILOT, CART, TDE, CLAGN
  - RRL, Mdwarf-flare, EB, uLens-Binary, uLens-Single (PyLIMA and GenLens), Cepheid, d-Sct, dwarf-nova
  - SNIIn-MOSFIT, SNIIn+HostXT_V19, SNII-NMF, SNII-Templates, SNII+HostXT_V19, SNIIb+HostXT_V19
  - SNIb-Templates, SNIb+HostXT_V19, SNIc-Templates, SNIc+HostXT_V19, SNIcBL+HostXT_V19
  - SNIa-SALT3, SNIa-91bg, SNIax
- Published taxonomies built on it: ORACLE uses 19 classes; NAPTIME uses 15 families; Fink/CATS uses
  broad classes. The classes are SNANA models mapped to the broker taxonomy (elasticc_origmap.txt
  in the GitHub repo, not read here).

**UCR** (timeseriesclassification.com; fetched with curl)
| Dataset | Train | Test | Length | Classes | Notes |
|---|---|---|---|---|---|
| ECG5000 | 500 | 4500 | 140 | 5 | BIDMC CHF record chf07; labels "obtained by automated annotation" |
| StarLightCurves | 1000 | 8236 | 1024 | 3 | "(purportedly) phase-aligned starlight curves"; donated by Keogh & Wei |
| Wafer | 1000 | 6164 | 152 | 2 | "10.7% of the train are abnormal, 12.1% of the test" |
| ElectricDevices | 8926 | 7711 | 96 | 7 | UK "Powering the Nation" household data |

Per-class counts for ECG5000 and StarLightCurves were not shown on these pages. Compute them from the
data and do not cite them from memory.

## Gaps (what this angle could not settle)
- No paper gives class-conditional (true-class) forecast coverage in astronomy. The sweep may have
  missed one, but none appeared in arXiv full-text queries such as conformal + light curve + supernova.
- No public prefix-to-future forecasting benchmark with a latent class exists for PLAsTiCC or
  ELAsTiCC. The protocol must be defined: cut times relative to first detection, the horizon grid,
  the bands, and DDF-only test.
- The ELAsTiCC taxonomy notebook on GitHub was not readable (GitHub API not enabled in this session).
- ICML/NeurIPS virtual pages were read only through curl. WebFetch was blocked for those hosts.
