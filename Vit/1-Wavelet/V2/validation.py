from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pywt
import numpy as np
from PIL import Image
import torch.nn as nn

class WaveletTransform:
    def __init__(self):  #construtor da classe - self=objeto
        self.img_size = 224
        self.wavelet = 'db4'
        self.levels = 2
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, pil_img):
        # PIL é a imagem original - transforma em Grayscale e float
        img_np = np.array(pil_img.convert('L')).astype(np.float32)
        
        #Wavelet
        coeffs = pywt.wavedec2(img_np, self.wavelet, mode='symmetric', level=self.levels)
        
        #salva as decomposições - pega o último nível
        LL = coeffs[0]
        LH, HL, HH = coeffs[1]
        
        # normaliza para 224x224
        def normalize_tensor(band):
            band_pil = Image.fromarray(((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8))
            band_resized = self.resize(band_pil)
            return transforms.ToTensor()(band_resized)
        
        bands = [
            normalize_tensor(LL),
            normalize_tensor(LH),  
            normalize_tensor(HL),
            normalize_tensor(HH)
        ]
        
        return torch.cat(bands, dim=0)

print(1)
# 1. Caminhos dos arquivos e diretórios
MODEL_DIR = "/home/pedro-furlan/Área de trabalho/checkpoint-V3.1-L1"
TEST_DATASET_DIR = "dataset/dataset_val"

print(2)
# 2. Processador e transformações (use as MESMAS do seu fine-tuning)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
transform = WaveletTransform()

print(3)
# 3. Dataset externo (pasta deve ter subpastas para cada classe)
test_dataset = ImageFolder(root=TEST_DATASET_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(4)
# 4. Carregar o modelo salvo do fine-tuning (ajuste os nomes dos labels se necessário)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(5)
# 5. Avaliação
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(6)
# 6. Métricas
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)

# Nomes das classes (ajuste conforme seu dataset)
class_names = test_dataset.classes  # Pega automaticamente do ImageFolder

# Visualização da matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('Vit/Wavelet/V2/confusion_matrix.png', dpi=300, bbox_inches='tight')

print(7)
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
