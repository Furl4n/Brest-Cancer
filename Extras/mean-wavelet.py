from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

train_ds = ImageFolder(
    root="/home/pedro-furlan/Área de trabalho/codigos/dataset_wavelet/dataset_train",
    transform=transforms.ToTensor()
)

loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)

pixel_count = 0
sum_pixels = torch.zeros(3)
sum_sq_pixels = torch.zeros(3)

for images, _ in loader:
    b, c, h, w = images.shape
    pixel_count += b * h * w
    sum_pixels += torch.sum(images, dim=[0, 2, 3])
    sum_sq_pixels += torch.sum(images ** 2, dim=[0, 2, 3])

mean = sum_pixels / pixel_count
std = torch.sqrt((sum_sq_pixels / pixel_count) - (mean ** 2))

print("mean =", mean.tolist())
print("std =", std.tolist())