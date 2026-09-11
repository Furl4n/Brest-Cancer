import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms

MEAN = [0.3639675974845886, 0.49822163581848145, 0.4940685033798218]
STD = [0.3144570589065552, 0.017851922661066055, 0.02476944774389267]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

ds = ImageFolder(
    root="/home/pedro-furlan/Área de trabalho/codigos/dataset_wavelet/dataset_train",
    transform=transform
)

img, label = ds[0]

mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)

img_vis = img * std + mean
img_vis = img_vis.clamp(0, 1)

plt.imshow(img_vis.permute(1, 2, 0).numpy())
plt.title(f"label={label}")
plt.axis("off")
plt.show()