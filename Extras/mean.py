import torch
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Corrige tamanho variável
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder('dataset_train', transform=transform)
dataset

loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

mean = 0.
std = 0.
nb_samples = 0.

for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean, std)
