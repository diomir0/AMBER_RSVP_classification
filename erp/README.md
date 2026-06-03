# ERP visualisation

This directory contains scripts for visualising Event-Related Potentials (ERPs)
on the AMBER EEG dataset, after the five denoising pipelines (RAW, ASR,
ICLabel, MARA, GEDAI) have been applied.

## `plot_erps.py`

The script operates in two modes:

### Grand-average mode (default)

For every denoising method × RSVP condition (X1, X2, X4, X6, X8), all
subjects/sessions are concatenated and the ERP figures show individual
trials (light) and their mean (bold).

Outputs are written to:

```
figures/
├── RAW/
│   ├── X1_rare_occipital.png          # grand-average (trials + mean)
│   ├── X1_rare_occipital_mean.png      # MNE Evoked.plot()
│   ├── X1_rare_occipital_trials.png    # MNE plot_image (GFP)
│   ├── ...
│   └── summary_occipital.png           # per-method summary grid
├── ASR/ ...
└── ...
```

### Single-recording mode (`--single`)

Each recording file (e.g. P01-Ss1-X1) is plotted separately so you can
inspect subject-level variability. Optionally narrow the selection with
`--subject` and/or `--session`.

Outputs are written to:

```
figures/<method>/single/
├── P01-Ss1-X1_rare_occipital.png
├── P01-Ss1-X1_rare_occipital_mean.png
├── P01-Ss1-X1_rare_occipital_trials.png
├── P01-Ss1-X1_frequent_occipital.png
├── ...
└── P01-Ss1_summary_occipital.png
```

### MNE dedicated methods used

* `mne.io.read_epochs_eeglab` — load the pre-epoched `.set` files.
* `mne.concatenate_epochs` — pool trials across subjects/sessions (grand-average).
* `mne.Epochs.average` — compute the mean ERP (`mne.Evoked`).
* `mne.Evoked.plot` — render the mean ERP with channel traces and confidence band.
* `mne.Epochs.plot_image` — render the per-trial GFP image.

### Usage

```bash
# Default: grand-average for every method and RSVP condition
python plot_erps.py

# Restrict to a single denoising method
python plot_erps.py --method RAW

# Restrict to a subset of RSVP tasks
python plot_erps.py --tasks X1 X2

# Use a different electrode cluster
python plot_erps.py --cluster central

# Plot a single channel (overrides --cluster)
python plot_erps.py --picks POz

# ── Grouped-task mode ───────────────────────────────────────────────

# Combine X1+X2 (standard RSVP) into a single ERP figure
python plot_erps.py --group-tasks --tasks X1 X2

# Combine X4+X6+X8 (artifact RSVP) into a single ERP figure
python plot_erps.py --group-tasks --tasks X4 X6 X8

# Both groups at once (standard + artifact) — produces separate
# per-group figures plus a cross-group summary
python plot_erps.py --group-tasks

# ── Single-recording mode ──────────────────────────────────────────

# Plot every recording individually
python plot_erps.py --single

# Only subject P01
python plot_erps.py --single --subject 1

# Only P01-Ss2
python plot_erps.py --single --subject 1 --session 2

# Single recording + grouped tasks: P01-Ss2, X1+X2 combined
python plot_erps.py --single --subject 1 --session 2 --group-tasks --tasks X1 X2
```