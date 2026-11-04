
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from octscreen.common.preprocess import resize_gray, to_chw_array


class OCTDataset(Dataset):
    """Load OCT images from the existing split folders."""

    def __init__(self, root, split, spec, class_names):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.root = Path(root) / split
        self.spec = dict(spec)
        self.class_names = list(class_names)

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {self.root}"
            )

        # Preserve the class order from the shared config.
        self.class_to_idx = {
            name: i for i, name in enumerate(self.class_names)
        }

        actual_classes = sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir()
        )

        if actual_classes != sorted(self.class_names):
            raise ValueError(
                f"Unexpected folders in {self.root}: "
                f"{actual_classes}"
            )

        self.samples = []

        valid_exts = {".jpeg", ".jpg", ".png"}

        for class_name in self.class_names:
            class_dir = self.root / class_name

            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in valid_exts:
                    self.samples.append(
                        (path, self.class_to_idx[class_name])
                    )

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

        self.labels = np.asarray(
            [label for _, label in self.samples],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        with Image.open(image_path) as img:
            # Shared preprocessing: grayscale + resize.
            img = resize_gray(img, self.spec)

        # Shared preprocessing: normalisation + CHW layout.
        arr = to_chw_array(img, self.spec)

        # Convert NumPy array to PyTorch tensor.
        x = torch.from_numpy(np.asarray(arr, dtype=np.float32))

        return x, int(label)
