from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from custom_data import MyImages

# training_data = MyImages(Path("../split.json"), "train")
# train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
#
# # Display image
# train_features = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# img = train_features[0].squeeze()
# plt.imshow(torch.permute(img, (1, 2, 0)))
# plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, X)  # autoencoder

        # Backpropagation
        optimizer.zero_grad()  # for not accumulating gradients from past iterations
        loss.backward()  # get gradients
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

