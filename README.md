# Predicting plant species and disease from leaf images (PlantVillage)

Code for the report on class-imbalance handling (weighted loss, focal loss, oversampling) for a
ResNet-18 on the Pepper, Potato and Tomato subset of PlantVillage (15 classes).

Runs on Google Colab with the project folder on Google Drive
(`PROJECT_ROOT` in the notebooks, `PROJECT_DIR` in `src/dataset.py`).
Put `plant_data.zip` (the PlantVillage images) in that folder. Checkpoints are not included.

## Files

| File | Content |
|---|---|
| `random_split.ipynb` | Unzips the data and makes the first random split |
| `src/Experiments_v2.ipynb` | Main notebook: data checks, leaf-grouped split, experiments, analyses, figures, final test evaluation |
| `src/dataset.py` | Data folders from `splits.json`, transforms, data loaders |
| `src/models.py` | ResNet-18 with a 15-class head |
| `src/losses.py` | Plain, weighted (E1a) and focal (E1b) loss |
| `src/trainer.py` | Training loop and evaluation |
| `src/experiment.py` | Runs one experiment for all seeds, saves results and the final test evaluation |
| `src/experiment_log.py` | Writes `runs/experiment_log.csv` |
| `src/sanity_checks.py` | Checks run before training |

## Order of `Experiments_v2.ipynb`

1. Setup, data loaders and sanity checks
2. Data checks and the leaf-grouped split (writes the final `splits.json`)
3. Run an experiment (set `CONFIG['experiment_id']`)
4. Analyses of the runs
5. Reduced training sets
6. E2 and background checks
7. Report figures
8. Final test evaluation

## Experiments

| ID | Meaning |
|---|---|
| E0, E0g, E0g2 | Earlier baselines |
| E0ga | Final baseline (cosine learning rate) |
| E1a / E1b / E1c | Weighted loss / focal loss / oversampling |
| E2 | Stronger augmentation |
| E0f25, E0f10, E0f05, E0f02 | Baseline on 25, 10, 5, 2 percent of the training data |

Seeds 0, 42 and 123, 10 epochs, batch size 32.

## Folders

| Folder | Content |
|---|---|
| `src/` | Modules and the notebooks above (`Experiments.ipynb` and `Setup_and_Modules.ipynb` are the first versions) |
| `runs/` | Results, analyses and `experiment_log.csv` |
| `frac_f02`, `frac_f05`, `frac_f10` | Reduced-data runs and the notebook that produced them |
| `figures/`, `Appendix figures/` | Report figures |
| `early_work/` | Course lab notebooks and the first data loader |
| `splits*.json`, `split_groups_*.csv` | Split files (`splits.json` is the final one) |
