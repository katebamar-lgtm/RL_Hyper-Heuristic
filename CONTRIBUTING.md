# Contributing

This repository accompanies a scientific study. Please preserve the distinction between reproducing the saved manuscript results and running a new experiment.

## Before changing code

- Do not overwrite `results/phase4_raw_runs.jsonl` or `data/lyon/case_study_raw_runs.jsonl` with newly generated runs.
- When a change affects objective evaluation, an operator, normalization, HV/IGD/GD, or statistical tests, describe its impact on reproducibility and update the manuscript-facing tables only after reviewing the regenerated outputs.
- Run the lightweight package checks and the paper-specific post-processing validator before submitting changes.
- Do not commit private credentials, local environment folders, cache files, or outputs from unrelated experiments.

## Validation

```bash
python -m compileall -q model scripts examples main.py
python scripts/postprocessing/validate_paper_consistency.py .
```

For a change to the post-processing pipeline, run the full reproduction command from `README.md` and then run the validator. The full pipeline may take several minutes.

## Licence and data

The project software is distributed under the MIT License; see [`LICENSE`](LICENSE). Contributors must ensure they have the rights to submit their contributions under that licence. Do not add material whose rights are incompatible with this licence without clarifying the applicable terms.

The MIT licence for the software does not automatically cover external or derived OSM/GTFS data. Check the provenance, attribution, and redistribution terms for the exact source data before adding or redistributing data files. Do not add third-party datasets unless their applicable terms permit the intended use.
