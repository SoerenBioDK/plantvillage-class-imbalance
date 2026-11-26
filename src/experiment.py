import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import build_dataloaders
from .experiment_log import append_run, code_version, make_row
from .losses import get_loss_function
from .models import get_resnet18
from .trainer import evaluate_full, fit

DEFAULTS = {
    "seeds": [0, 42, 123],
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-4,
    "img_size": 224,
    "num_classes": 15,
    "freeze_backbone": False,
    "dropout": 0.0,
    "lr_schedule": "constant",  # "constant" or "cosine" (decays to 0 over the epochs)
}

RESULT_KEYS = ("accuracy", "macro_f1", "loss", "class_report", "confusion_matrix", "preds", "targets")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(
    config,
    project_root,
    device,
    eval_test=False,
    notes="",
    train_dir="/content/data/train",
    val_dir="/content/data/val",
    test_dir="/content/data/test",
    save_weights=True,
):
    """Train one experiment for every seed in the config.

    Per seed it saves the history, validation results and (optionally) test results to
    runs/<experiment>.json, appends a row to runs/experiment_log.csv and stores the weights in
    checkpoints/. Seeds that are already saved are skipped, so an interrupted run can be resumed.
    """
    cfg = {**DEFAULTS, **config}
    project_root = Path(project_root)
    runs_dir = project_root / "runs"
    ckpt_dir = project_root / "checkpoints"
    runs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    exp = cfg["experiment_id"]
    out_path = runs_dir / f"{exp}.json"
    csv_path = runs_dir / "experiment_log.csv"
    src_dir = project_root / "src"

    saved = (
        json.load(open(out_path)) if out_path.exists() else {"config": cfg, "results": [], "classes": None}
    )
    done = {r["seed"] for r in saved["results"]}
    print(f"{exp}: seeds already finished: {sorted(done)}")

    for seed in cfg["seeds"]:
        if seed in done:
            print(f"--- skipping {exp} seed {seed} (already saved) ---")
            continue
        print(f"\n--- running {exp} | seed {seed} | code version {code_version(src_dir)} ---")
        set_seed(seed)
        train_loader, val_loader, test_loader, class_names = build_dataloaders(
            train_dir, val_dir, test_dir, cfg
        )
        model = get_resnet18(
            cfg["num_classes"], freeze_backbone=cfg["freeze_backbone"], dropout=cfg["dropout"]
        ).to(device)
        criterion = get_loss_function(exp, train_loader, device=device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])
        if cfg["lr_schedule"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"], eta_min=0.0
            )
        elif cfg["lr_schedule"] == "constant":
            scheduler = None
        else:
            raise ValueError(f"unknown lr_schedule: {cfg['lr_schedule']}")

        t0 = time.time()
        history, val = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            cfg["epochs"],
            class_names=class_names,
            scheduler=scheduler,
        )
        runtime = time.time() - t0

        test = None
        if eval_test:
            test = evaluate_full(model, test_loader, device, criterion=criterion, class_names=class_names)
            print(
                f"--> TEST (final evaluation only): acc {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f}"
            )

        record = {
            "seed": seed,
            "code_version": code_version(src_dir),
            "runtime_s": round(runtime, 1),
            "history": history,
            "val": {k: val[k] for k in RESULT_KEYS},
            "test": {k: test[k] for k in RESULT_KEYS} if test else None,
        }
        saved["results"].append(record)
        saved["classes"] = class_names
        saved["config"] = cfg
        # save after every seed so a disconnected Colab session loses nothing
        with open(out_path, "w") as f:
            json.dump(saved, f)
        append_run(csv_path, make_row(cfg, seed, history, val, runtime, src_dir, class_names, test, notes))
        if save_weights:
            torch.save(model.state_dict(), ckpt_dir / f"{exp}_seed{seed}.pt")
        print(
            f"saved seed {seed}: val acc {val['accuracy']:.4f}, val macro-F1 {val['macro_f1']:.4f}, "
            f"{runtime / cfg['epochs']:.1f}s per epoch"
        )
    return saved


def final_test_eval(
    config,
    project_root,
    device,
    seeds=None,
    train_dir="/content/data/train",
    val_dir="/content/data/val",
    test_dir="/content/data/test",
):
    """Final evaluation of finished runs on the test set, using the saved checkpoints.

    Run once per chosen model, after all decisions have been made on the validation set.
    """
    cfg = {**DEFAULTS, **config}
    project_root = Path(project_root)
    exp = cfg["experiment_id"]
    _, _, test_loader, class_names = build_dataloaders(train_dir, val_dir, test_dir, cfg)
    out = []
    for seed in seeds or cfg["seeds"]:
        model = get_resnet18(
            cfg["num_classes"], freeze_backbone=cfg["freeze_backbone"], dropout=cfg["dropout"]
        ).to(device)
        model.load_state_dict(
            torch.load(project_root / "checkpoints" / f"{exp}_seed{seed}.pt", map_location=device)
        )
        res = evaluate_full(model, test_loader, device, class_names=class_names)
        print(f"{exp} seed {seed}: TEST acc {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f}")
        out.append(
            {
                "seed": seed,
                **{
                    k: res[k]
                    for k in ("accuracy", "macro_f1", "class_report", "confusion_matrix", "preds", "targets")
                },
            }
        )
    with open(project_root / "runs" / f"{exp}_test.json", "w") as f:
        json.dump({"config": cfg, "classes": class_names, "results": out}, f)
    return out
