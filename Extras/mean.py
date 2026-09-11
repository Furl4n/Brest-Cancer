import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("dataset_orientado/dataset_train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

pixel_count = 0
sum_pixels = torch.zeros(3)
sum_sq_pixels = torch.zeros(3)

for data, _ in loader:
    b, c, h, w = data.shape
    pixel_count += b * h * w
    sum_pixels += data.sum(dim=[0, 2, 3])
    sum_sq_pixels += (data ** 2).sum(dim=[0, 2, 3])

mean = sum_pixels / pixel_count
std = torch.sqrt(sum_sq_pixels / pixel_count - mean ** 2)

print(mean)
print(std)