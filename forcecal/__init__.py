"""forcecal: estimation of distal tip forces from a proximal F/T sensor.

Analysis code for the repeated calibration runs of a handheld orthopaedic
instrument with a Resense HEX-32 six-axis sensor, calibrated against a
Force Dimension Lambda.7 haptic device.

Modules, in the order of the analysis:

    settings     read and check configs/analysis.yaml and configs/objects.yaml
    constants    CSV column names, unit convention, frames S and EE, model list
    data         load runs, check them, reduce each dwell to one point
    models       object geometry and the eight force-reconstruction models
    evaluation   train/test splits, model fitting and scoring, dead-weight check
    metrics      model-free quantities (ISO 376, drift) and error metrics
    campaign     statistics across the repeated remountings
    diagnostics  hysteresis, coupling matrix, anchor drift, apparent EI
    plots        all figures
    report       Markdown/LaTeX reports and run_info.json

The two commands in the repository folder (results_report.py and
diagnostics_report.py) call these modules.
"""

__version__ = "1.0.0"
