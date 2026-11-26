import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "date",
    "experiment",
    "seed",
    "code_version",
    "epochs",
    "batch_size",
    "lr",
    "img_size",
    "train_loss_final",
    "val_loss_final",
    "val_acc",
    "val_macro_f1",
    "val_min_class_recall",
    "val_min_recall_class",
    "test_acc",
    "test_macro_f1",
    "runtime_s",
    "sec_per_epoch",
    "notes",
    "config_json",
]


def code_version(src_dir):
    """Short hash of all .py files in src/, so every logged run can be tied to the code that produced it."""
    h = hashlib.md5()
    for p in sorted(Path(src_dir).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def append_run(csv_path, row):
    """Append one run to the CSV log (the file is created with a header if needed)."""
    csv_path = Path(csv_path)
    new_file = not csv_path.exists()
    full = {c: row.get(c, "") for c in COLUMNS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(full)


def make_row(config, seed, history, val, runtime_s, src_dir, class_names, test=None, notes=""):
    report = val["class_report"]
    recalls = {c: report[c]["recall"] for c in class_names if c in report}
    worst = min(recalls, key=recalls.get) if recalls else ""
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": config["experiment_id"],
        "seed": seed,
        "code_version": code_version(src_dir),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "img_size": config["img_size"],
        "train_loss_final": round(history[-1]["train_loss"], 5),
        "val_loss_final": round(history[-1]["val_loss"], 5),
        "val_acc": round(val["accuracy"], 5),
        "val_macro_f1": round(val["macro_f1"], 5),
        "val_min_class_recall": round(recalls[worst], 5) if worst else "",
        "val_min_recall_class": worst,
        "test_acc": round(test["accuracy"], 5) if test else "",
        "test_macro_f1": round(test["macro_f1"], 5) if test else "",
        "runtime_s": round(runtime_s, 1),
        "sec_per_epoch": round(runtime_s / config["epochs"], 1),
        "notes": notes,
        "config_json": json.dumps(config, sort_keys=True),
    }


def summarise(csv_path, experiments=None):
    """Mean and standard deviation over seeds per experiment, from the CSV log (validation metrics)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if experiments:
        df = df[df["experiment"].isin(experiments)]
    g = df.groupby("experiment", sort=False)
    out = g.agg(
        runs=("seed", "count"),
        val_acc_mean=("val_acc", "mean"),
        val_acc_std=("val_acc", "std"),
        val_f1_mean=("val_macro_f1", "mean"),
        val_f1_std=("val_macro_f1", "std"),
        min_class_recall_mean=("val_min_class_recall", "mean"),
        sec_per_epoch=("sec_per_epoch", "mean"),
    )
    return out.round(4)
