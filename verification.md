# Verification

This file separates three kinds of check. None of them refits the published models.

## A. Scores recomputed from the released tables

From the repository root:

```bash
python -m src.report_archived_scores
python -m src.report_block_interval
```

`report_archived_scores` keeps the 3,210 profiles whose recorded layers cover 0–100 cm and prints the archived primary-model error and the archived random-forest error.

`report_block_interval` replays the seed-11 block bootstrap. It uses 2,000 draws. The sampling unit is the spatial block, not the profile. The generator is advanced in the original order: the 0–30 cm contrasts first, then the 0–100 cm contrasts. The script compares each paired interval with `outputs/metrics/round11_coverage.json`. The 0–100 cm random-forest contrast has a mean difference of 0.232 kg m−2 and a 95% interval of 0.021–0.497. A match is a recomputation from the released tables. It is not a new model fit.

The planned comparison uses the archived random-forest predictions. `outputs/tables/round12_interval_profiles.csv` is a later calibration refit. Every row has `is_planned_contrast` set to false. Its random-forest column must not be used to recompute the paired difference.

## B. Steps that need data outside this archive

Training, covariate extraction and geographic figures also require:

- the Zenodo version-10 profile table
- the standard administrative boundaries with map review number GS (2024) 0650
- a Google Earth Engine extraction of the covariates

Those files are not in this archive. The map review number belongs in the figure caption. It is not drawn on the map.

## C. What cannot be reproduced bit for bit

The network weights were not saved. A later primary-model refit agreed with the archived point predictions, and that agreement is not a recovery of the weights. The later random-forest refit does not match the archived forest. The score scripts do not rebuild the mean rasters.
