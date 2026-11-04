# octscreen/data/dataset.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from octscreen.common.config import to_dict
from octscreen.common.preprocess import resize_gray, to_chw_array
from octscreen.data.discover import build_filename_index


def load_split(cfg, split):
    df = pd.read_csv(cfg.data.splits_csv, dtype={"patient_id": str})
    df = df[df["split"] == split]
    limit = OmegaConf.select(cfg, "data.limit_per_class", default=None)
    if limit:                                   # quick experiments: cap images per class
        df = df.sample(frac=1.0, random_state=int(cfg.split.seed)).groupby("label", group_keys=False).head(int(limit))
    return df.reset_index(drop=True)


def resolve_paths(df, cfg):
    root = Path(cfg.data.root)
    if OmegaConf.select(cfg, "data.path_resolution", default="relpath") == "filename":
        index = build_filename_index(root, list(cfg.data.image_exts))
        missing = [f for f in df["filename"] if f not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} files not found under {root}, e.g. {missing[:3]}")
        return [index[f] for f in df["filename"]]
    return [root / p for p in df["relpath"]]


class OCTDataset(Dataset):
    """Returns (tensor[3,H,W], label_idx). Uses the SHARED preprocessing (contract C1)."""

    def __init__(self, paths, labels, spec, augment=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.spec = dict(spec)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = resize_gray(im, self.spec)          # step 1 (shared)
        if self.augment is not None:
            img = self.augment(img)                   # train only
        x = torch.from_numpy(to_chw_array(img, self.spec))  # step 2 (shared)
        return x, int(self.labels[i])


def make_dataset(cfg, split, augment=None):
    df = load_split(cfg, split)
    classes = list(cfg.data.class_names)
    labels = df["label"].map({c: i for i, c in enumerate(classes)})
    if labels.isna().any():
        raise ValueError(f"Labels not in class_names: {sorted(set(df.loc[labels.isna(), 'label']))}")
    return OCTDataset(resolve_paths(df, cfg), labels.to_numpy(), to_dict(cfg.preprocess), augment), df
