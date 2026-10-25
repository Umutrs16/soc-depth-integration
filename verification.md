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

The equal-area hidden-layer comparison is a separate within-profile calculation. `outputs/metrics/round16_eaqss.json` stores its profile-mean errors, the thickness groups, the density and relative errors, and the clip-at-zero sensitivity. The primary summary remains the unclipped spline at lambda 0.1. Refitting it requires the processed layer table, which is not in this archive. It does not replace the spatial-block scores, and it is not an adaptive sampling trial or a prospective sampling budget.

`src.round17_intervals` can be rerun from the archived out-of-fold stocks and the coverage table. Its Gaussian scales and interval scores do not replace the percentile interval. A wider interval is not an information gain. The buffer and biome numbers in `outputs/metrics/round17_spatial.json` come from a new forest fit. They are not the archived forest and not a retraining of the primary network. The buffer was not applied to the neural model. Rebuilding that fit needs the processed layers and the RESOLVE ecoregion polygons, which are not in this archive. `outputs/metrics/round17_layer_subsets.json` is a retrospective omission of layers that were already recorded. The two designs are different profile sets. `outputs/metrics/round18_common_cohort.json` pairs the shared profiles and can still compare different omitted layers. It is not an active sampling trial. `outputs/metrics/round17_wosis.json` summarises public WoSIS horizons inside the standard polygons. The complete-stock subset has 35 profiles, 27 of them in Taiwan, and none matches a version-10 coordinate at four decimal degrees. Every scored stock layer uses one bulk-density code. The concentration summary is in g/kg and is not a stock. `outputs/tables/round18_wosis_anonymous_profiles.csv` has per-profile absolute errors and no coordinates or WoSIS identifiers. The 27 Taiwan profiles are summarised separately from the other eight, and no significance test is reported for those eight. The raw snapshot is not in this archive. These horizons are not a centimetre-scale reference and not a spatial-block score. `outputs/metrics/round19_eaqss_groups.json` adds signed bias on the published thickness bands and on two midpoint groups. It is computed from the saved layer table and does not refit the spline. The common-cohort file is a retrospective omission sensitivity. The same profiles can still have different hidden layers, so it is not a same-target sampling comparison.

## B. Steps that need data outside this archive

Training, covariate extraction and geographic figures also require:

- the Zenodo version-10 profile table
- the standard administrative boundaries with map review number GS (2024) 0650
- a Google Earth Engine extraction of the covariates
- the WoSIS December 2023 snapshot and the RESOLVE Ecoregions 2017 polygons, for the external horizon check and the biome refit

Those files are not in this archive. The map review number belongs in the figure caption. It is not drawn on the map. WoSIS rows are not redistributed.

## C. What cannot be reproduced bit for bit

The network weights were not saved. A later primary-model refit agreed with the archived point predictions, and that agreement is not a recovery of the weights. The later random-forest refit does not match the archived forest. The score scripts do not rebuild the mean rasters.
