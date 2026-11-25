# Temporal Graph Learning for IoT Network Digital Twins

This repository contains the code for the article:

> V. Ravi, S. Yamaguchi, M. A. B. Ahmadon, V. Ganesalingam and M. H. B. Kamilin, *Temporal Graph Learning for IoT Network Digital Twins: A Robustness-Tested Comparison with History-Based Baselines for Communication-Graph Prediction*, submitted to *Sensors* (MDPI).

The study predicts the next 300 s communication graph of an IoT network from the previous four. It compares seven learned temporal graph models with six simple history methods on two public IoT traffic datasets. The learned models are adapted re-implementations of published architectures.

The code does the following:

- builds directed communication graphs from packet captures;
- trains the learned models with five seeds;
- selects operating points on validation data;
- compares learned models and simple methods with a two-level paired block bootstrap;
- runs every pre-registered and exploratory analysis reported in the article.

The analysis plans are included and fixed by SHA-256. The scripts check these hashes before they run.

## Repository layout

```
.
├── preprocess_pcaps_clean.py        packet captures -> one directed graph per 300 s window
├── build_graph_dataset.py           chronological splits, node features, L = 4 histories
├── check_reprocess.py               checks that re-processing a capture reproduces its graphs
├── process_unsw_newdays.bat         processes the 12 further UNSW capture days
├── train_dyngem_style_predictor_corrected.py      DynGEM (adapted)
├── train_dyngraph2vec_aernn_style_corrected.py    dyngraph2vec AERNN (adapted)
├── train_vgrnn_predictor_corrected.py             VGRNN (adapted)
├── train_tgae_style_predictor_corrected.py        TGAE (adapted); asymmetric TGAE is a decoder variant
├── train_vrdag_style_predictor_corrected.py       VRDAG (adapted)
├── train_static_vgae_updated.py                   Static VGAE
├── figures/
│   └── search_progress.py           figure: progress of the Bayesian optimisation
└── ndt_stage5/                      training, evaluation and every analysis stage
    ├── train_seeds.py               trains a model with seeds 42-46 and scores it at every operating point
    ├── evaluate_pilot.py            data checks, measures, simple history methods, threshold selection
    ├── evaluate_temporal.py         inference for the temporal models
    ├── run_analysis_seeds.py        paired two-level block bootstrap with Holm adjustment
    ├── history_lr.py                History-LR
    ├── stage6_analysis.py           edges per window, new-edge categories, History-LR, calibration
    ├── check_reciprocity.py         two-way edges and the ceiling of symmetric decoders
    ├── ...                          further scripts, listed by stage below
    ├── stage7a.bat ... stage10_steps.bat   the exact commands of each stage
    └── *.md                         analysis plans and records of deviations
```

## Requirements

The results were produced on Windows with an NVIDIA RTX 4060 Laptop GPU, using:

- Python 3.12.7
- PyTorch 2.7.1 (CUDA 12.8)
- NumPy 1.26.4
- Optuna 5.1.0 (optimisation stages only)

The other packages are listed in `requirements.txt`. Install PyTorch for your CUDA version from <https://pytorch.org> first, then:

```
pip install -r requirements.txt
```

The simple methods and the statistical analyses run on the CPU.

## Data

The datasets are not redistributed here. Download them from their providers:

- **UNSW IoT Traces** (Sivanathan et al., IEEE TMC 2019): <https://iotanalytics.unsw.edu.au/iottraces.html>
- **Deakin IoT Traffic** (Pasquini et al., IEEE Data Descriptions 2025): <https://doi.org/10.26187/deakin.28013234>

Use the daily packet captures of the following days:

| Split | Training | Validation | Test |
|---|---|---|---|
| UNSW, main | 23-27 Sep 2016 | 8 Oct 2016 | 9-10 Oct 2016 |
| UNSW, second split | 23-25 Sep 2016 | 26 Sep 2016 | 27 Sep 2016 |
| UNSW, further test days | (main split) | (main split) | 28 Sep-7 Oct and 11-12 Oct 2016 |
| Deakin, main | 30-31 Mar 2024 | 4 Apr 2024 | 21 Apr 2024 |
| Deakin, second split | 30 Mar 2024 | 31 Mar 2024 | 4 Apr 2024 |

## Reproducing the results

The scripts expect this folder to be the project root, with the analysis scripts in `ndt_stage5/`. Run every command from `ndt_stage5/` unless stated otherwise; outputs are written next to it, in the project root. Most steps skip work that has already finished, so they can be re-run after an interruption.

### 1. Communication graphs (from the project root)

```
python preprocess_pcaps_clean.py --pcap_dir <folder with the .pcap files> --output_dir outputs --windows 300
```

This writes `outputs/graphs_300s/<day>/graph_snapshots.pkl`. A packet is kept when both addresses are private (`ipaddress.ip_address(...).is_private`); multicast and limited-broadcast packets are removed.

### 2. Datasets (from the project root)

```
python build_graph_dataset.py --graph_root outputs/graphs_300s --output_dir dataset_ready/300s ^
  --train_dates 16-09-23 16-09-24 16-09-25 16-09-26 16-09-27 --val_dates 16-10-08 --test_dates 16-10-09 16-10-10
```

The Deakin dataset is built in the same way into `dataset_ready_deakin_chronological/300s`, with day identifiers of the form `IoT_2024-03-30`. The second splits are built by `ndt_stage5/make_unsw_fold2.py` and `ndt_stage5/make_deakin_fold2.py` (see `--help`).

### 3. Main analysis (from `ndt_stage5/`)

```
python train_seeds.py --project-root .. --dataset unsw --out ..\seeds_unsw
python train_seeds.py --project-root .. --dataset unsw --models tgae_bias tgae_bilinear --out ..\seeds_unsw_decoder
python history_lr.py --project-root .. --dataset unsw --out ..\history_lr_unsw
python run_analysis_seeds.py --project-root .. --dataset unsw --runs ..\seeds_unsw --out ..\analysis_unsw
python stage6_analysis.py --project-root .. --dataset unsw --runs ..\seeds_unsw ..\seeds_unsw_decoder --history-lr ..\history_lr_unsw --out ..\stage6_unsw
```

Replace `unsw` by `deakin`, `unsw_fold2` or `deakin_fold2` for the other splits. `tgae_bilinear` is asymmetric TGAE and `tgae_bias` is TGAE with a bias term.

### 4. Further stages

| Stage (article section) | Scripts | Commands |
|---|---|---|
| Pre-registered confirmations on the second splits (4.4) | `make_deakin_fold2.py`, `make_unsw_fold2.py`, `evaluate_preregistration.py`, `evaluate_prereg_unsw.py` | `--help` of each script |
| Robustness round: tuning, window length, horizon, history length, class weighting (4.5) | `tune_models.py`, `run_scope.py`, `make_merged_windows.py`, `make_horizon_dataset.py`, `horizon_support.py`, `compare_best.py`, `summarize_stage7.py` | `stage7a.bat` to `stage7e.bat` |
| Periodicity check (4.6) | `periodicity_check.py` | `--help` |
| Twelve further UNSW days (4.7) | `process_unsw_newdays.bat` (project root), `make_unsw_newdays.py`, `score_newdays.py`, `confirm_newdays.py` | `stage8.bat` |
| Training versus validation performance (4.8) | `capacity_check.py` | `--help`; also run in `stage9_steps.bat` |
| Bayesian optimisation (4.9) | `bo_search.py`, `bo_analysis.py` | `stage9_steps.bat` |
| Random-search check (4.9) | `bo_search.py --sampler random`, `rs_compare.py` | `stage10_steps.bat` |
| Pair-history analysis (4.10) | `data_analysis.py` | `python data_analysis.py --project-root .. --out ..\data_analysis` |

The batch files contain the exact commands used for the article. Run them from `ndt_stage5/` in a terminal where the Python environment is active.

## Analysis plans

Each plan was written before the runs it governs. The scripts record its SHA-256 hash, and the optimisation scripts refuse to run if the plan has changed. The plans are in `ndt_stage5/` and are also reproduced in the article's Supplementary Materials.

| File | Stage | SHA-256 (first 16 characters) |
|---|---|---|
| `preregistration_deakin_fold2.md` | Deakin confirmation | `24ed95daa7ad7eb7` |
| `preregistration_unsw_fold2.md` | UNSW confirmation | `c36e1e8d9cd2babe` |
| `deviations_unsw_fold2.md` | UNSW confirmation, deviation record | `d8af5d3fe0180428` |
| `preregistration_stage7.md` | Robustness round | `bf0382497655eafe` |
| `preregistration_stage7_addendum_7C.md` | Robustness round, horizon addendum | `abd928f2fb0496d2` |
| `deviations_stage7.md` | Robustness round, deviation record | `05810bed10fac395` |
| `periodicity_check_plan.md` | Periodicity check | `0d299956f2c88b7a` |
| `preregistration_stage8_newdays.md` | Further UNSW days | `3454f6a94c7e171d` |
| `preregistration_stage9_optimisation.md` | Bayesian optimisation | `35eea43286b32822` |
| `preregistration_stage10_random_search.md` | Random-search check | `7205eaebd0c4ba20` |
| `deviations_stage10.md` | Correction to the random-search plan | `705b1e761038d33c` |
| `data_analysis_plan.md` | Pair-history analysis (exploratory) | `fc694ea51759d269` |

On Windows, check a hash with `certutil -hashfile <file> SHA256`.

## Citation

If you use this code, please cite the article. The full reference will be added here when it is published.
