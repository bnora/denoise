import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import wandb

import src.training_fashionMNIST as trainer
from src.model import Autoencoder


wandb.init(project="fashionMNIST")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


training_data = datasets.FashionMNIST(
    root="fashionMNIST_data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="fashionMNIST_data",
    train=False,
    download=True,
    transform=ToTensor()
)

# learning_rate = 1e-3
# batch_size = 64
# epochs = 5

wandb.config = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 5,
}

train_dataloader = DataLoader(training_data, batch_size=wandb.config["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=wandb.config["batch_size"], shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}, type {train_features.type()}")

model = Autoencoder(base_channel_size=16,
                    latent_dim=16,
                    num_input_channels=1)
model.to(device)

# pred = model(train_features)
# print(f"Output batch shape: {pred.size()}")
# # Display image and label.
# img = train_features[0].detach().numpy().squeeze()
# img_out = pred[0].detach().numpy().squeeze()
# print(np.shape(img_out), np.shape(img))
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")
# fig.add_subplot(1, 2, 2)
# plt.imshow(img_out, cmap="gray")
# plt.show()
# print(f"Label: {label}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

for n_epoc in range(wandb.config["epochs"]):
    print(f"Epoch {n_epoc + 1}\n-------------------------------")
    trainer.train_loop(train_dataloader, model, loss_fn, optimizer)
    trainer.test_loop(test_dataloader, model, loss_fn)
print("Done!")


# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")


