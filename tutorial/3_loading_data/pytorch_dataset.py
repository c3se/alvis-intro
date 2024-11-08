from zipfile import ZipFile
from io import BytesIO
from fnmatch import fnmatch
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define the path to the dataset
PATH_TO_DATASET = '/mimer/NOBACKUP/Datasets/tiny-imagenet-200/tiny-imagenet-200.zip'

def examine_zipfile_structure(path_to_dataset: str):
    """Prints the number and names of entries in the zip file."""
    with ZipFile(path_to_dataset, 'r') as datazip:
        print(f"Number of entries in the zipfile {len(datazip.namelist())}")
        print(*datazip.namelist()[:7], "...", *datazip.namelist()[-3:], sep="\n")

def read_labels_from_txt(path_to_dataset: str):
    """Reads and returns the labels from wnids.txt."""
    with ZipFile(path_to_dataset, "r") as datazip:
        return datazip.read("tiny-imagenet-200/wnids.txt").decode("utf8").split()

def count_train_images(path_to_dataset: str):
    """Returns the number of training images in the zip file."""
    with ZipFile(path_to_dataset) as datazip:
        return len([fn for fn in datazip.namelist() if 'train' in fn and fn.endswith('.JPEG')])

def visualize_sample_images(path_to_dataset: str):
    """Visualizes a grid of sample images from the training set."""
    fig, ax_grid = plt.subplots(3, 3, figsize=(15, 15))
    with ZipFile(path_to_dataset) as datazip:
        filenames = [fn for fn in datazip.namelist() if 'train' in fn and fn.endswith('.JPEG')]
        for ax, fn in zip(ax_grid.flatten(), filenames):
            label = fn.split("/")[-1].split('_')[0]
            img = plt.imread(BytesIO(datazip.read(fn)), format="jpg")
            ax.imshow(img)
            ax.set_title(f'Label {label}')
    fig.tight_layout()
    plt.show()

# Construct a Dataset class for our dataset
class TinyImageNetDataset(Dataset):
    def __init__(self, path_to_dataset: str, split: str):
        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split, select 'train', 'val' or 'test'.")
        if split in ["val", "test"]:
            raise NotImplementedError("Only train split is currently implemented.")
        
        self.zfpath = path_to_dataset
        # Avoid reusing the file handle created here, for known issue with multi-worker:
        # https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
        self.zf = None
        with ZipFile(self.zfpath) as zf:
            # Get images from split
            self.imglist = [
                path for path in zf.namelist()
                if split in path and path.endswith(".JPEG")
            ]

            # Get look-up dictionary for word name ID to label
            wnids = zf.read("tiny-imagenet-200/wnids.txt").decode("utf8").split()
            self.wnid2label = {wnid: label for label, wnid in enumerate(wnids)}

    def get_label(self, path: str) -> int:
        word_name_id = path.split("/")[-1].split('_')[0]
        return self.wnid2label[word_name_id]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.zf is None:
            self.zf = ZipFile(self.zfpath)

        # Convert image to Tensor of size (Channel, Px, Py)    
        imgpath = self.imglist[idx]
        img_array = np.array(Image.open(BytesIO(self.zf.read(imgpath))))
        if img_array.ndim < 3:
            # Greyscale to RGB
            img_array = np.repeat(img_array[..., np.newaxis], 3, -1)

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        # Get label from filename
        label = self.get_label(imgpath)
        return img_tensor, label

