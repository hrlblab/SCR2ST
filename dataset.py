# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class STDataset(Dataset):
    """
    ST Dataset: each record must contain:
        - sample_id
        - patch_id
        - expression
        Optional:
        - sc_embed (ST embedding for RL sampling)
        - position (coordinates)
    """

    def __init__(self, records, patch_root, transform=None):
        self.records = records
        self.patch_root = patch_root
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # load patch
        img_path = os.path.join(
            self.patch_root, rec["sample_id"], f"{rec['patch_id']}.png"
        )
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # expression (log transform)
        expr = torch.from_numpy(np.log1p(rec["expression"].astype(np.float32)))

        # Used for RL sampling
        emb = rec.get("sc_embed", None)
        if emb is not None:
            emb = torch.from_numpy(emb.astype(np.float32))

        pos = rec.get("position", None)
        if pos is not None:
            pos = torch.tensor(pos, dtype=torch.float32)

        return img, expr, emb, pos
