import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate(model, dataloader, device):
    """Returns (top-1 accuracy, macro-F1, per-class report)."""
    out = evaluate_full(model, dataloader, device)
    return out["accuracy"], out["macro_f1"], out["class_report"]


def evaluate_full(model, dataloader, device, criterion=None, class_names=None):
    """Evaluate on one loader. Returns metrics, per-class report, confusion matrix and raw predictions."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if criterion is not None:
                total_loss += criterion(outputs, targets).item() * inputs.size(0)
            n += inputs.size(0)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    labels = list(range(len(class_names))) if class_names is not None else None
    names = list(class_names) if class_names is not None else None
    result = {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "loss": (total_loss / n) if criterion is not None else None,
        "class_report": classification_report(
            all_targets, all_preds, labels=labels, target_names=names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=labels).tolist(),
        "preds": all_preds,
        "targets": all_targets,
    }
    return result


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    class_names=None,
    verbose=True,
    scheduler=None,
):
    """Train for a fixed number of epochs and record the history on the validation set.

    The model from the last epoch is returned (no best-epoch selection). val_loss is always plain
    cross-entropy so that it is comparable across experiments. The scheduler is stepped once per epoch.
    """
    eval_criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        lr_used = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate_full(model, val_loader, device, criterion=eval_criterion, class_names=class_names)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "epoch_time_s": round(time.time() - t0, 1),
            "lr": lr_used,
        }
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | lr {lr_used:.2e} | train loss {train_loss:.4f} | "
                f"val loss {val['loss']:.4f} | val acc {val['accuracy']:.4f} | val F1 {val['macro_f1']:.4f} "
                f"| {row['epoch_time_s']}s"
            )
    return history, val
