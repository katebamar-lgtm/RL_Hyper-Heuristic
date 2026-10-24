# Reinforcement Learning and Adaptive Operator Selection for Multi-Objective Optimization in Multimodal Transportation Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- After the Zenodo upload, add: [![DOI](https://zenodo.org/badge/DOI/YOUR-DOI.svg)](https://doi.org/YOUR-DOI) -->
Archived version: DOI `[DOI]`

Research code and saved experimental outputs for the manuscript of the same title.

The project studies adaptive operator selection (AOS) for the **four-objective multimodal multi-criteria shortest-path problem (MCSP)**, minimizing travel time, distance, monetary cost, and number of transfers. Two architectures are implemented and compared with static and uniform-random controls:

- **RL-MOEA/D**: MOEA/D with an online, context-dependent UCB-type bandit that selects among three reproduction strategies.
- **OMF-RL**: Optimization by Morphological Filters (OMF) with an offline-trained, frozen contextual policy that selects between the width and depth neighborhoods.

Five additional reference algorithms (NSGA-II, SPEA2, MOGWO, MOFA, MOABC) are included, giving 11 conditions on nine synthetic instances (20–120 nodes) plus an exploratory case study on a reduced, data-derived Lyon network.

> **Summary of findings.** RL-MOEA/D has a higher mean hypervolume than static MOEA/D, but the difference is not significant after Holm correction and it does not outperform random selection. The tested OMF-RL policy is significantly worse than both OMF controls; at deployment it behaves as a constant "always-depth" choice. The experiments do not establish a general benefit from learning-based operator selection. See the manuscript for details and limitations.

## Quick start: reproduce the manuscript tables

**The manuscript's tables do not require rerunning the optimization algorithms.** The repository contains the raw objective fronts from the final campaign. The post-processing pipeline recomputes all indicators and statistics from them and takes less than a minute on a standard laptop.

```bash
# 1. Python 3.11 or 3.12 is recommended (the GitHub workflow uses 3.12)
python -m pip install -r requirements-postprocessing.txt

# 2. Recompute indicators and statistics from the stored fronts
python scripts/postprocessing/run_full_postprocessing.py .

# 3. Check the regenerated results against the manuscript values
python scripts/postprocessing/validate_paper_consistency.py .
```

Expected final line: `ALL CHECKS PASSED. No optimization runs were required.`

The pipeline reads `results/phase4_raw_runs.jsonl` and `data/lyon/case_study_raw_runs.jsonl` and writes only to `postprocessing_reports/`. It never executes an optimizer and never overwrites the raw runs.

### Where each manuscript table comes from

| Manuscript item | File in `postprocessing_reports/step3_statistics/` |
|---|---|
| Table 6 (HV, 11 algorithms, mean ranks) | `synthetic_all_algorithm_HV_by_scale.csv`, `synthetic_friedman_mean_ranks.csv` |
| Tables 7–8 (six primary conditions: indicators, effort) | `synthetic_six_primary_conditions_by_scale.csv` |
| Table 9 (primary Wilcoxon tests, Holm) | `synthetic_primary_HV_instance_level.csv` |
| Median sensitivity (Section 6.3) | `synthetic_primary_HV_instance_median_sensitivity.csv` |
| Secondary IGD/GD tests | `synthetic_secondary_IGD_GD_instance_level.csv` |
| Friedman / Nemenyi / 55-pair Wilcoxon–Holm | `synthetic_friedman_summary.csv`, `synthetic_nemenyi_pairs.csv`, `synthetic_allpairs_wilcoxon_holm.csv` |
| Pooled repetition-level sensitivity | `synthetic_pooled_run_sensitivity.csv` |
| Table 11 (Lyon) | `lyon_all_algorithms_by_route.csv`, `lyon_route_level_HV_comparisons.csv` |
| All headline numbers in text form | `paper_reproduction_report.txt` |

## Indicator and statistical protocol

- Four objectives are minimized: travel time, distance, monetary cost, and transfers.
- For each instance, min–max bounds are shared across all stored algorithm fronts and repetitions.
- Hypervolume (HV) uses the reference point `(1.1, 1.1, 1.1, 1.1)` and 20,000 deterministic Monte Carlo sample locations shared by all algorithms on the same instance. HV is therefore not restricted to `[0, 1]`: the reference-box volume is `1.1^4 = 1.4641`.
- IGD and GD use empirical reference fronts (not known Pareto fronts) and are compared **within** a family only (MOEA/D variants; OMF variants). They must not be ranked across different `reference_policy` groups.
- Primary inference: the unit of analysis is the instance (indicator averaged over repetitions); exact two-sided Wilcoxon signed-rank tests; Holm correction across four prespecified HV contrasts.
- Secondary endpoints: IGD and GD, four contrasts each, Holm correction across eight tests.
- Exploratory analyses: 11-algorithm Friedman test with Nemenyi post-hoc test (CD = 5.03), and a 55-pair Wilcoxon–Holm family (limited power with nine instances).
- Pooled repetition-level tests are sensitivity analyses only, because repetitions are nested within instances.
- Lyon uses three route-level means and is interpreted descriptively.

### Naming conventions in the data files

Some saved files use legacy labels. The correspondence with the manuscript is:

| Data files | Manuscript |
|---|---|
| `OMF-statique` | OMF-static |
| `OMF-aleatoire` | OMF-random |
| `cpu_seconds` | Elapsed wall-clock time in seconds (measured with `time.time()`; the field name is historical) |

## Final algorithm configurations

Parameters used in the final campaign (Table 5 of the manuscript). They are defined in `model/benchmark_runner.py`; the calibration records are in `calibration/`.

| Algorithm | Final parameters |
|---|---|
| NSGA-II | pc = 0.9, pm = 0.2 |
| SPEA2 | pc = 0.7, pm = 0.1, archive = 0.5 × N |
| MOEA/D | T = 8, pc = 0.9, pm = 0.2 |
| RL-MOEA/D | T = 12, UCB coefficient c = 0.02, rich reward |
| RL-MOEA/D-random | T = 16 |
| MOGWO | mutation rate = 0.15 |
| MOFA | β₀ = 1.5, γ = 1.0, α₀ = 0.5 |
| MOABC | limit = 5 |
| OMF-static | NF = 20, IT = 20, NN = 8, R = 0.25, weak dominance |
| OMF-random | NF = 20, IT = 20, NN = 8, weak dominance, uniform choice |
| OMF-RL | NF = 20, IT = 20, NN = 8, weak dominance, frozen offline policy |

Population size is 30, 50 and 80 for small, medium and large instances, and the maximum evaluation budget is 300 × population size (9,000, 15,000 and 24,000). OMF-based conditions stop earlier because of their restart/iteration/neighbor schedule. The conditions differ in several calibrated settings (for example T), so the comparison is between calibrated configurations, not a controlled ablation of learning.

## Repository contents

| Path | Purpose |
|---|---|
| `model/` | Network representation, objective evaluation, host algorithms, and RL controllers |
| `model/rl/` | Offline OMF-RL training, state representation, and evaluation utilities |
| `calibration/` | Saved calibration records and the frozen OMF-RL policy |
| `data/lyon/` | Processed Lyon graph files and saved case-study runs/indicators (see `data/lyon/NOTICE.txt`) |
| `results/phase4_raw_runs.jsonl` | Raw synthetic fronts of the final campaign (1,650 runs) |
| `results/phase4_indicators.jsonl` | Recomputed synthetic indicator records, used for consistency checks |
| `postprocessing_reports/step2_recomputed/` | Recomputed indicators, shared normalization manifest, and report |
| `postprocessing_reports/step3_statistics/` | Statistical tables and paper-reproduction report |
| `scripts/postprocessing/` | Indicator recomputation, statistical analysis, and validation scripts |
| `examples/run_phase4_batch.py` | Optional new synthetic runs; output defaults to `results/reproduction/` |
| `requirements-postprocessing.txt` | Pinned dependencies for the post-processing pipeline |
| `requirements.txt` | Dependencies for the algorithm code and visualization utilities |
| `CITATION.cff` | Citation metadata |

## Running new optimization experiments

New runs are **not required** to reproduce the manuscript and produce a different experiment. The batch runner writes to `results/reproduction/` by default to protect the included raw data.

```bash
python -m pip install -r requirements.txt
python examples/run_phase4_batch.py final-S1 20 --out results/reproduction/phase4_raw_runs.jsonl
```

Do not append new runs to `results/phase4_raw_runs.jsonl` or `data/lyon/case_study_raw_runs.jsonl` unless you are deliberately creating a new, documented campaign. A quick smoke test of all 11 algorithms on one small instance is available with `python Test-Algos.py`.

## Lyon case study: data provenance and reuse

The Lyon graph is a **reduced, data-derived network** (696 node identifiers: 80 OSM-derived and 616 GTFS-prefixed; 25 transit lines). It is an exploratory case study, not a complete representation of the city's network. All transit fares are empty in the source data and are treated as zero, and the access links are direct walking shortcuts.

- The repository includes the processed graph files and the saved final runs. They are sufficient to reproduce the reported analysis.
- The archive does **not** preserve the exact GTFS provider, feed URL, version or retrieval date, nor the exact OSM extraction date and query extent. The graph therefore cannot be rebuilt from the original external sources. Do not assume that a currently available feed is the one used in this study.
- Parts of the data are derived from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0). The licence of the original GTFS feed has not been verified.

See [`data/lyon/NOTICE.txt`](data/lyon/NOTICE.txt) for the attribution and licensing notice. Check the terms of the original sources before redistributing these data independently of the code.

## Environment and security note

The post-processing pipeline was tested with the versions pinned in `requirements-postprocessing.txt` (NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0). No API credentials are required.

`calibration/final_omf_rl_policy.pkl` is a Python pickle containing the frozen OMF-RL policy. Pickle files can execute code when loaded: open this file only from a trusted copy of this repository.

## License

The software is released under the [MIT License](LICENSE). The licence applies to the code. It does not determine the rights applicable to the OpenStreetMap- and GTFS-derived data in `data/lyon/`; see `data/lyon/NOTICE.txt`.

## Citation

If you use this code, please cite the manuscript and the archived version of this repository (see [`CITATION.cff`](CITATION.cff)).

```bibtex
@software{kateb_hachemi_amar_hyperheuristic_2026,
  author  = {Kateb Hachemi Amar, Amar and Hamdadou, Djamel and Abbou, Sarra Itidal and Benbrik, Nihad and Bouazdia, Rania},
  title   = {Code and experimental outputs for ``Reinforcement Learning and Adaptive
             Operator Selection for Multi-Objective Optimization in Multimodal
             Transportation Networks''},
  year    = {2026},
  version = {1.0.0},
  doi     = {[DOI]},
  url     = {[URL]}
}
```

Replace `[URL]` and `[DOI]` after the GitHub upload and the Zenodo archiving, and update the citation of the manuscript when publication details are available.

## Contact

Corresponding author: Amar Kateb Hachemi Amar, Hassiba Benbouali University of Chlef, Algeria (`a.kateb@univ-chlef.dz`).
