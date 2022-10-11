import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision import io


class MyImages(Dataset):
    def __init__(self, json_file: Path, set_name: str):
        """
        Args:
            json_file: Path to the split file
            set_name: train, val, or test
        """
        self.json_file = json_file
        with open(json_file, "r") as in_file:
            data_split = json.load(in_file)  # reads into dict
        self.image_list = data_split.get(set_name, None)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: Union[int, torch.IntTensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.read_image(self.image_list[idx])

        return image
