"""Sanity checks run before training.

A. check_data         : per-channel statistics in all splits and a look at images and labels
B. check_split        : no overlap between splits, class counts per split, class-order check
C. check_initial_loss : loss of the untrained model compared with the expected value ln(C)
D. overfit_one_batch  : can the model memorise one small batch?

run_all_checks() runs A to D and prints a summary table.
"""

import copy
import hashlib
import math
from pathlib import Path

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- A: data
def _channel_stats(loader, n_samples=1000, seed=0):
    """Per-channel mean/std over a random subset (val and test loaders are not shuffled)."""
    from torch.utils.data import DataLoader, Subset

    ds = loader.dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[: min(n_samples, len(ds))].tolist()
    sub = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, num_workers=0)
    n, s, ss = 0, torch.zeros(3), torch.zeros(3)
    lo, hi = float("inf"), float("-inf")
    for x, _ in sub:
        x = x.float()
        s += x.sum(dim=(0, 2, 3))
        ss += (x**2).sum(dim=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
        lo, hi = min(lo, x.min().item()), max(hi, x.max().item())
    mean = s / n
    std = (ss / n - mean**2).clamp(min=0).sqrt()
    return mean.tolist(), std.tolist(), lo, hi


def check_data(
    loaders,
    class_names,
    n_samples=1000,
    n_show=8,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    show=True,
    save_path=None,
):
    """loaders: dict like {"train": train_loader, "val": val_loader, "test": test_loader}."""
    per_split = {}
    for name, loader in loaders.items():
        m, s, lo, hi = _channel_stats(loader, n_samples)
        per_split[name] = (m, s)
        print(
            f"{name:>5}: mean per channel {[round(v, 3) for v in m]}, std per channel "
            f"{[round(v, 3) for v in s]}, min {lo:.2f}, max {hi:.2f}"
        )
    ok_range = all(
        max(abs(v) for v in m) < 0.5 and 0.6 < min(s) and max(s) < 1.4 for m, s in per_split.values()
    )
    names = list(per_split)
    ok_same = all(
        max(abs(a - b) for a, b in zip(per_split[names[0]][0], per_split[n][0])) < 0.1 for n in names[1:]
    )
    ok = ok_range and ok_same
    print(f"A. data statistics sensible: {ok_range}, splits consistent with each other: {ok_same}")

    if show:
        import matplotlib.pyplot as plt

        x, y = next(iter(loaders["train"]))
        m_t = torch.tensor(mean).view(1, 3, 1, 1)
        s_t = torch.tensor(std).view(1, 3, 1, 1)
        imgs = (x[:n_show] * s_t + m_t).clamp(0, 1)
        fig, axes = plt.subplots(1, n_show, figsize=(2.0 * n_show, 2.6))
        for ax, img, lab in zip(axes, imgs, y[:n_show].tolist()):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(class_names[lab][:18], fontsize=7)
            ax.axis("off")
        plt.suptitle("Training images with labels (de-normalised). Check by eye that labels fit the images.")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.show()
    return ok


# ---------------------------------------------------------------- B: split
def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_split(datasets, hash_files=True):
    """datasets: dict like {"train": ImageFolder, "val": ImageFolder, "test": ImageFolder}.

    Checks that the class lists are identical, that no file is shared between splits (by relative path,
    file name and content hash) and prints the number of images per class and split.
    Different photos of the same physical leaf are not detected here.
    """
    names = list(datasets)
    classes = [datasets[n].classes for n in names]
    same_classes = all(c == classes[0] for c in classes)
    print(f"B. identical class lists and order in all splits: {same_classes}")

    keys = {}
    for n in names:
        ds = datasets[n]
        root = Path(ds.root)
        keys[n] = {
            "path": {Path(p).relative_to(root).as_posix() for p, _ in ds.samples},
            "name": {Path(p).name for p, _ in ds.samples},
            "hash": {_md5(p) for p, _ in ds.samples} if hash_files else set(),
        }
    overlap_total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            o = {k: len(keys[a][k] & keys[b][k]) for k in ("path", "name", "hash")}
            overlap_total += o["path"] + o["name"] + o["hash"]
            print(
                f"   overlap {a}/{b}: same path {o['path']}, same file name {o['name']}, "
                f"same content {o['hash'] if hash_files else 'not checked'}"
            )

    import numpy as np

    n_cls = len(classes[0])
    counts = {n: np.bincount([lab for _, lab in datasets[n].samples], minlength=n_cls) for n in names}
    total = {n: int(counts[n].sum()) for n in names}
    grand = sum(total.values())
    print("   split sizes: " + ", ".join(f"{n} {total[n]} ({total[n] / grand:.1%})" for n in names))
    header = f"   {'class':<46}" + "".join(f"{n:>8}" for n in names) + "  all"
    print(header)
    for k in range(n_cls):
        row = [int(counts[n][k]) for n in names]
        print(f"   {classes[0][k][:45]:<46}" + "".join(f"{v:>8}" for v in row) + f"  {sum(row)}")
    allc = sum(counts[n] for n in names)
    print(f"   imbalance ratio (largest / smallest class, all data): {allc.max() / allc.min():.1f}x")
    ok = same_classes and overlap_total == 0
    print(f"B. split check passed: {ok}")
    return ok


# ---------------------------------------------------------------- C: initial loss
@torch.no_grad()
def check_initial_loss(model, loader, criterion, device, num_classes, n_batches=10):
    """The loss of the untrained model should be close to ln(C) (2.708 for 15 classes).

    With a pretrained backbone it is usually a bit higher. NaN or a much larger value points to a bug.
    """
    model = copy.deepcopy(model).to(device).eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item() * x.size(0)
        n += x.size(0)
    loss = total / n
    expected = math.log(num_classes)
    ok = math.isfinite(loss) and 0.5 * expected < loss < expected + 1.5
    print(
        f"C. initial loss {loss:.3f}, expected about ln({num_classes}) = {expected:.3f} "
        f"(a bit higher is normal with a pretrained backbone). Passed: {ok}"
    )
    return ok, loss


# ---------------------------------------------------------------- D: overfit one batch
def overfit_one_batch(
    model, loader, criterion, device, steps=100, lr=1e-4, batch_size=32, show=True, save_path=None
):
    """Train a copy of the model on one fixed batch. Loss should go to about 0 and accuracy to 1.0."""
    x, y = next(iter(loader))
    x, y = x[:batch_size].to(device), y[:batch_size].to(device)
    m = copy.deepcopy(model).to(device).train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    losses, accs = [], []
    for _ in range(steps):
        opt.zero_grad()
        out = m(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((out.argmax(1) == y).float().mean().item())
    ok = losses[-1] < 0.05 and accs[-1] == 1.0
    print(
        f"D. overfit one batch of {len(y)}: final loss {losses[-1]:.4f}, final accuracy {accs[-1]:.2f}. Passed: {ok}"
    )
    if show:
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
        a1.plot(losses)
        a1.set_yscale("log")
        a1.set_title("Loss on one batch")
        a1.set_xlabel("step")
        a2.plot(accs)
        a2.set_ylim(-0.02, 1.02)
        a2.set_title("Accuracy on the same batch")
        a2.set_xlabel("step")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.show()
    return ok, losses, accs


# ---------------------------------------------------------------- unit tests
def test_focal_equals_ce():
    """FocalLoss with gamma = 0 must equal cross-entropy, and gamma = 2 must be smaller."""
    import torch.nn as nn

    from .losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(64, 15)
    targets = torch.randint(0, 15, (64,))
    ce = nn.CrossEntropyLoss()(logits, targets)
    f0 = FocalLoss(gamma=0.0)(logits, targets)
    f2 = FocalLoss(gamma=2.0)(logits, targets)
    ok = torch.allclose(ce, f0, atol=1e-6) and f2 < ce
    print(
        f"Focal loss test: CE {ce.item():.4f}, focal gamma=0 {f0.item():.4f}, focal gamma=2 {f2.item():.4f}. Passed: {ok}"
    )
    return ok


def test_class_weight_order(train_loader, class_names, device="cpu"):
    """The class weights of E1a must be indexed in the same order as class_names / model outputs."""
    import numpy as np

    from .losses import get_loss_function

    crit = get_loss_function("E1a", train_loader, device=device)
    counts = np.bincount([lab for _, lab in train_loader.dataset.samples], minlength=len(class_names))
    w = crit.weight.cpu().numpy()
    smallest = int(counts.argmin())
    ok = int(w.argmax()) == smallest and len(w) == len(class_names)
    print(
        f"Class weight order test: largest weight is for '{class_names[int(w.argmax())]}', smallest class is "
        f"'{class_names[smallest]}'. Passed: {ok}"
    )
    return ok


# ---------------------------------------------------------------- run everything
def run_all_checks(loaders, datasets, model, criterion, device, class_names, show=True, hash_files=True):
    """Runs A to D and prints a summary table."""
    num_classes = len(class_names)
    res = {}
    res["A"] = check_data(loaders, class_names, show=show)
    res["B"] = check_split(datasets, hash_files=hash_files)
    res["C"], init_loss = check_initial_loss(model, loaders["train"], criterion, device, num_classes)
    res["D"], losses, accs = overfit_one_batch(model, loaders["train"], criterion, device, show=show)
    print("\n| Check | Result | Pass? |\n|---|---|:---:|")
    labels = {
        "A": "A. Data statistics and images",
        "B": "B. Split overlap and class order",
        "C": f"C. Initial loss ({init_loss:.3f} vs ln(C) = {math.log(num_classes):.3f})",
        "D": f"D. Overfit one batch (final loss {losses[-1]:.4f}, acc {accs[-1]:.2f})",
    }
    for k in "ABCD":
        print(f"| {labels[k]} | see output above | {'yes' if res[k] else 'NO'} |")
    return res
