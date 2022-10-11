from pathlib import Path

from torch.utils.data import DataLoader

from src.data import MyImages


def test_dataset(test_data_dir):
    batch_size = 1
    training_data = MyImages(Path(test_data_dir) / "split.json", "train")
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    train_features = next(iter(train_dataloader))
    image_height, image_width = 2304, 3072
    assert train_features.size() == (batch_size, 3, image_height, image_width)
    assert train_features[0, 0, 100, 100] != 0  # There is some red value in that pixel

    # for manual testing uncomment this
    # img = train_features[0].squeeze()
    # plt.imshow(torch.permute(img, (1, 2, 0)))
    # plt.show()