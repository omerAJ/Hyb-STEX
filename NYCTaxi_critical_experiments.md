# NYCTaxi Critical Experiments

| Experiment | Avg MAE In | Avg MAE Out |
| --- | ---: | ---: |
| fix phase-wise training evs_90 final (5 seeds) | 11.5020 | 9.2640 |
| nodewise soft-gpd tail correction final (3 seeds) | 11.8433 | 9.6367 |
| best old-base tail-only before joint refine (5 seeds) | 11.5441 | 9.3958 |
| promoted best static_then_joint old-base GPD (5 seeds) | 11.3499 | 9.1795 |
| GPD utility ablation: normal_excess (5 seeds) | 11.3923 | 9.2073 |
| GPD utility ablation: point_excess (5 seeds) | 11.4283 | 9.2271 |
| ablation: remove phase-1 corrected_mae influence (5 seeds) | 11.3975 | 9.2240 |
