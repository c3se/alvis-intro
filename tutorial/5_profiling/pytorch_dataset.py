import zipfile
import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class TinyImageNetDataset(Dataset):
    def __init__(self, path_to_dataset: str, split: str):
        if split not in ["train", "val"]:
            raise ValueError("Invalid split, select 'train' or 'val'.")
        
        self.zfpath = path_to_dataset
        self.zf = None
        with zipfile.ZipFile(self.zfpath) as zf:
            # Get images for the specified split
            self.imglist = [
                path for path in zf.namelist()
                if split in path and path.endswith(".JPEG")
            ]

            # Create a mapping from word name IDs to labels
            wnids = zf.read("tiny-imagenet-200/wnids.txt").decode("utf-8").split()
            self.wnid2label = {wnid: label for label, wnid in enumerate(wnids)}

            if split == "val":
                # Parse validation set annotations
                self.filename2wnid = {}
                val_annotations = zf.read("tiny-imagenet-200/val/val_annotations.txt").decode("utf-8").splitlines()
                for line in val_annotations:
                    fname, wnid, *_ = line.split("\t")
                    self.filename2wnid[fname] = wnid

    def get_label(self, path: str) -> int:
        # Get label based on filename
        if hasattr(self, 'filename2wnid'):  # For validation set
            wnid = self.filename2wnid[os.path.basename(path)]
        else:  # For training set
            wnid = path.split("/")[-1].split('_')[0]
        return self.wnid2label[wnid]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zfpath)

        imgpath = self.imglist[idx]
        img_array = np.array(Image.open(self.zf.open(imgpath)))
        if img_array.ndim < 3:
            img_array = np.repeat(img_array[..., np.newaxis], 3, -1)

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        label = self.get_label(imgpath)
        return img_tensor, label
