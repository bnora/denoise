import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

import src.custom_data


def test_dataset(test_data_dir):
    batch_size = 1
    training_data = src.custom_data.MyImages(Path(test_data_dir) / "split.json", "train")
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    train_features = next(iter(train_dataloader))
    image_height, image_width = 2304, 3072
    assert train_features.size() == (batch_size, 3, image_height, image_width)
    assert train_features[0, 0, 100, 100] != 0  # There is some red value in that pixel

    # for manual testing uncomment this
    # img = train_features[0].squeeze()
    # plt.imshow(torch.permute(img, (1, 2, 0)))
    # plt.show()


test_splits = [((50, 50, 0), (1, 1, 0)),
               ((100, 0, 0), (2, 0, 0)),
               ((0, 100, 0), (0, 2, 0)),
               ((150, 0, 150), (1, 0, 1))]


@pytest.mark.parametrize("split, expected", test_splits)
def test_get_dataset_split(split, expected, test_data_dir):
    my_split = src.custom_data.get_dataset_split(test_data_dir, split)
    print(my_split.train)
    assert len(my_split.train) == expected[0]
    assert len(my_split.val) == expected[1]
    assert len(my_split.test) == expected[2]


def test_save_split_as_json(test_data_dir, tmp_path):
    my_split = src.custom_data.get_dataset_split(test_data_dir, (50, 50, 0))
    src.custom_data.save_split_as_json(my_split, tmp_path)
    assert os.path.exists(tmp_path / "split.json")  # exists
    assert os.path.getsize(tmp_path / "split.json")  # is not empty

