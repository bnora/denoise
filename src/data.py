import json
import random
from pathlib import Path
from typing import Union, Tuple
import dataclasses


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


@dataclasses.dataclass
class DataSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def get_dataset_split(data_dir: Path, split: Tuple) -> DataSplit:
    """ Loops over all jpg images in the directory, and splits them randomly into train, val and test sets"""
    list_of_images = [f for f in Path(data_dir).iterdir() if f.suffix == '.JPG']
    random.shuffle(list_of_images)
    n_images = len(list_of_images)

    split = [x/sum(split) for x in split]
    n_training = int(n_images * split[0])
    n_validation = int(n_images * split[1])
    n_test = n_images - n_training - n_validation
    print(
        f"Splitting {n_images} images into {n_training} training, {n_validation} validation and {n_test} test samples")
    my_split = DataSplit([str(x) for x in list_of_images[0:n_training]],
                         [str(x) for x in list_of_images[n_training:n_training + n_validation]],
                         [str(x) for x in list_of_images[n_training + n_validation:]])
    return my_split


def save_split_as_json(my_split: DataSplit, out_dir: Path) -> None:
    out_file = out_dir / "split.json"
    json_object = json.dumps(dataclasses.asdict(my_split), indent=4)
    with open(out_file, 'w') as f_out:
        f_out.write(json_object)