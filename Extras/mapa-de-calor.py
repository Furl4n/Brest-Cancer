import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ==========================================
# 1. Configurações de Caminhos e Modelo
# ==========================================
model_path = "Vit/Vit_final_stage2/checkpoint-1940"
# Substitua pelo caminho de uma imagem do seu dataset de validação
image_path = "dataset_orientado/dataset_val/benigno/0362/C_0362_1.RIGHT_CC.jpg"

def generate_vit_attention_map(model_path, img_path):
    # Carrega o processador e o modelo com output_attentions=True
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        model_path, 
        output_attentions=True
    )
    model.eval()

    # Prepara a imagem
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # Inferência
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions  # Tupla com atenções de todas as camadas

    # Pega a predição para o título
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    labels = model.config.id2label
    conf = probs[0][pred_idx].item()

    # ==========================================
    # 2. Processamento da Matriz de Atenção
    # ==========================================
    # Pegamos a última camada de atenção (índice -1)
    # Shape: (batch, heads, patches+1, patches+1) -> (1, 12, 197, 197)
    last_layer_att = attentions[-1][0]
    
    # Média entre todas as 'heads' de atenção
    avg_att = torch.mean(last_layer_att, dim=0)

    # O ViT usa o token [CLS] (índice 0) para classificar.
    # Queremos ver a atenção que o [CLS] deu para os outros 196 patches (14x14).
    cls_attention = avg_att[0, 1:].reshape(14, 14)
    
    # Normalização para exibição (0 a 1)
    cls_attention = cls_attention.numpy()
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())

    # ==========================================
    # 3. Visualização Lado a Lado
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Imagem Original
    ax1.imshow(img.resize((224, 224)))
    ax1.set_title(f"Original\nPredição: {labels[pred_idx]} ({conf*100:.2f}%)")
    ax1.axis('off')

    # Mapa de Atenção sobreposto
    ax2.imshow(img.resize((224, 224)))
    # 'jet' é ótimo para calor, 'interpolation=bilinear' suaviza os quadrados
    ax2.imshow(cls_attention, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax2.set_title("Mapa de Atenção (Onde a IA focou)")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Executa a função
try:
    generate_vit_attention_map(model_path, image_path)
except Exception as e:
    print(f"Erro ao processar: {e}")