from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print(1)
# 1. Caminhos dos arquivos e diretórios
MODEL_DIR = "/home/pedro-furlan/Área de trabalho/checkpoint-4400"
TEST_DATASET_DIR = "dataset_limpo/dataset_val"

print(2)
# 2. Processador e transformações (as mesmas do fine-tuning)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2048, 0.2048, 0.2048], std=[0.2048, 0.2048, 0.2048])
])

print(3)
# 3. Dataset externo (pasta deve ter subpastas para cada classe)
test_dataset = ImageFolder(root=TEST_DATASET_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(4)
# 4. Carregar o modelo salvo do fine-tuning (ajuste os nomes dos labels se necessário)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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
plt.savefig('Vit/2-final_tuning/confusion_matrix.png', dpi=300, bbox_inches='tight')

print(7)
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
