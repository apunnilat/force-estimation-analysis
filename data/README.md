# Data

The measurement data is not stored in this repository (about 30 MB per run).

**Location of the thesis dataset:** <add the polybox / ETH Research Collection link here>

## Expected folder layout

`configs/analysis.yaml` points to the data with `data.root` (default
`data/instrument_data`, relative to the repository folder) and lists the run
folders in `data.runs`:

```
data/instrument_data/
├── test_50/
│   ├── ground_truth.csv         required: calibration log (CSV schema 7)
│   ├── alignment_log.csv        optional: checked for alignment failures
│   └── weight_validation.csv    optional: dead-weight check
├── test_51/
│   └── ...
└── test_57/
```

Each run folder is one mounting of the instrument; the sensor was remounted
between runs. File names may carry suffixes (`ground_truth (2).csv`); the
shortest matching name is used.

## File formats

- `ground_truth.csv`: written by the Lambda.7 controller. Comment lines starting
  with `#` hold metadata (`schema_version`, `pattern_seed`, `ema_alpha`, ...).
  Columns are described in [`docs/conventions.md`](../docs/conventions.md).
- `weight_validation.csv`: columns `t_rel_s`, `stage` (`baseline_pre`, `loaded`,
  `baseline_post`) and the six `*_meas_EE_*` sensor channels; metadata
  `# mass_kg,<value>`.
- `alignment_log.csv`: any text log; lines containing `alignment_failed` mark
  the run in the integrity table.
