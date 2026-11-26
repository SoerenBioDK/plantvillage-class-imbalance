import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

PROJECT_DIR = Path("/content/drive/MyDrive/Søren og Magnus/Deep Learning /Project_deep")
ZIP_PATH = PROJECT_DIR / "plant_data.zip"
MANIFEST_PATH = PROJECT_DIR / "splits.json"

DATA_ROOT = Path("/content/data")
BASE_DIR = DATA_ROOT / "PlantVillage"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_data_extracted():
    """Unzip the dataset and build the train/val/test folders from splits.json if they are missing."""
    if not all(d.exists() and any(d.iterdir()) for d in (TRAIN_DIR, VAL_DIR, TEST_DIR)):
        print("Data not found locally. Extracting plant_data.zip to /content/data/ ...")

        if not BASE_DIR.exists():
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(DATA_ROOT)

        with open(MANIFEST_PATH) as f:
            split = json.load(f)

        for split_name, target_dir in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
            for rel_path in split[split_name]:
                src = BASE_DIR / rel_path
                dst = target_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dst))

        print("Data extraction done.")


def get_transforms(exp_id: str, img_size: int = 224):
    """E2 uses stronger augmentation; all other experiments only flip horizontally."""
    if exp_id == "E2":
        train_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        train_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    eval_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(train_dir: str, val_dir: str, test_dir: str, config: dict):
    ensure_data_extracted()

    train_tf, eval_tf = get_transforms(config["experiment_id"], config["img_size"])

    train_dataset = ImageFolder(train_dir, transform=train_tf)
    val_dataset = ImageFolder(val_dir, transform=eval_tf)
    test_dataset = ImageFolder(test_dir, transform=eval_tf)

    sampler = None
    shuffle = True

    # E1c: oversampling, every class is drawn equally often
    if config["experiment_id"] == "E1c":
        class_counts = np.bincount([label for _, label in train_dataset.samples])
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=shuffle, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.classes
