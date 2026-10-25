# Depth-integrated soil carbon profiles

Analysis code for a spatial-block evaluation of whether numerically consistent depth integration identifies a soil organic carbon profile. The manuscript and the Word-generation script are not part of this archive.

## Recompute the published comparison

From this directory:

```bash
python -m src.report_archived_scores
python -m src.report_block_interval
```

`report_archived_scores` reads the archived out-of-fold predictions and the union-coverage table. It keeps the 3,210 profiles whose recorded layers cover 0–100 cm and prints the primary-model error and the archived random-forest error. It does not train a model.

`report_block_interval` replays the seed-11 block bootstrap (2,000 draws; the sampling unit is the spatial block) and checks the paired intervals against the archived coverage file. See `verification.md` for what can be recomputed, what needs the source tables, and what cannot be reproduced bit for bit.

`src.round16_eaqss` fits the Bishop equal-area spline to hidden interior layers. The summary is in `outputs/metrics/round16_eaqss.json`, including thickness groups and the clip-at-zero sensitivity. The primary summary remains the unclipped spline at lambda 0.1. Refitting requires the processed layer table, which is not in this archive. The result is not a spatial score, not an adaptive sampling trial, and not a prospective sampling budget.

The planned comparison uses those archived random-forest predictions. `outputs/tables/round12_interval_profiles.csv` is a later calibration refit. Every row has `is_planned_contrast` set to false. Its random-forest column does not reproduce the archived forest and must not be used to recompute the paired difference.

## What is included

- Processing, model, calibration and figure modules in `src/`
- The archived prediction table and the union-coverage table
- The labelled calibration refit, kept separate from the planned comparison
- Figure images exported from those results
- `environment_pin.txt`, the versions used for the archived run, and `requirements.txt`

## What is not included

Profile tables, administrative boundaries and network weights are not in this archive. See NOTICE.md. Figure scripts that draw maps need the official boundary data and the processed profile table, so those maps cannot be regenerated from this archive alone. `src.experiment` and `src.mapping` are the original fitting procedures. Running them would replace the archived predictions and the mean surfaces; they are not required for the score script above, and the published scores were not produced by a fresh run of either module.

A match between a later primary-model refit and the archived point predictions is not a bit-level recovery of the unsaved weights.
