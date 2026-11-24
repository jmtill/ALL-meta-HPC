# Papers

PDFs were attached in chat on 2026-11-23 (not stored in the repo). Keep them somewhere you can re-attach.

## Meta-learning (Phase 17)

| Paper | PDF in hand | Model file | Status |
|---|---|---|---|
| Tighineanu et al., "Scalable Meta-Learning with Gaussian Processes" (ScaML-GP), arXiv 2312.00742 | yes, full with appendix | `scaml.py` | designed, written, tested |
| Fan, Han & Wang, "HyperBO+: Pre-training a universal prior for BO with hierarchical GPs", arXiv 2212.10538 | yes, full | `hyperprior_transfer.py` | not started; would reuse `fit_task_gp` |
| Rothfuss, Fortuin, Josifoski & Krause, "PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees", ICML 2021 | yes, full | `pacoh.py` | designed, written, tested on a synthetic environment (`PAPER_CARDS/pacoh.md`) |
| Nguyen, Low & Jaillet, "Learning to Learn with Gaussian Processes" (GPML), UAI 2021 | main paper only (supplement missing) | none | last or drop: predicts with a neural network, no GP predictive variance |

## Heteroskedastic (Phase 16)

| Paper | PDF in hand | Model file | Status |
|---|---|---|---|
| Binois, Gramacy & Ludkovski, "Practical Heteroscedastic GP Modeling for Large Simulation Experiments", JCGS 2018 | arXiv preprint 1611.05902v2 read; journal PDF not attached | `most_likely.py` / hetGP | first Phase 16 model. Phase 15's `hetgp_practical` already runs it through hetGPy |
| Wang, Chen & Chen, "A Variational Inference-Based Heteroscedastic GP Approach for Simulation Metamodeling", ACM TOMACS 29(1), 2019 | yes | `variational.py` | the replicate-aware variational model |
| Wang & Chen, "Distributed Variational Inference-Based Heteroscedastic GP Metamodeling", WSC 2019 | yes | – | distributed part probably unneeded at our data size |
| Lázaro-Gredilla & Titsias, "Variational Heteroscedastic Gaussian Process Regression", ICML 2011 | yes | `variational.py` | the original variational model |
| PyMC example "Heteroskedastic Gaussian Processes" (Goertz, updated by Krapu) | web page | `chained.py` reference | fully Bayesian reference; slow (≈3,300 s on 200 points with NUTS) |

## Scaffold placeholders no paper covers yet

Kersting 2007 (most-likely), Saul 2016 (chained), Snelson 2004 (warped), RGPE (Feurer 2018), FSBO (Wistuba & Grabocka
2021). Keep or delete: undecided.
