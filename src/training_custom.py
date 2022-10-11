from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from custom_data import MyImages

training_data = MyImages(Path("../split.json"), "train")
train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

# Display image
train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
img = train_features[0].squeeze()
plt.imshow(torch.permute(img, (1, 2, 0)))
plt.show()


