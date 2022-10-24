from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from src.custom_data import MyImages
from src.model import Autoencoder
import src.training_custom as trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

training_data = MyImages(Path("../split.json"), "train")
val_data = MyImages(Path("../split.json"), "val")

learning_rate = 1e-3
batch_size = 4
epochs = 5

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}, type {train_features.type()}")

model = Autoencoder(base_channel_size=16,
                    latent_dim=16,
                    num_input_channels=3)
model.to(device)
train_features=train_features.to(device)
print(f"Feature batch shape: {train_features.size()}, type {train_features.type()}, device {train_features.device}")
pred = model(train_features)
print(f"Output batch shape: {pred.size()}, type {pred.type()}")

# Display image and label.
rows = 4
fig = plt.figure()
for irow in range(rows):
    img = train_features[irow].detach().numpy().squeeze()
    img = np.transpose(img, axes=(1, 2, 0))  # image is [0, 1]
    fig.add_subplot(rows + 1, 2, 2*irow + 1)
    plt.imshow(img)
    img_out = pred[irow].detach().numpy().squeeze()
    img_out = np.transpose(img_out, axes=(1, 2, 0))  # image is [0, 1]
    fig.add_subplot(rows + 1, 2, 2*irow + 2)
    plt.imshow(img_out, cmap="gray")
plt.show()

#
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for n_epoc in range(epochs):
#     print(f"Epoch {n_epoc + 1}\n-------------------------------")
#     trainer.train_loop(train_dataloader, model, loss_fn, optimizer)
#     trainer.test_loop(val_dataloader, model, loss_fn)
# print("Done!")


# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")


