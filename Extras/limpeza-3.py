import cv2
import numpy as np
import os
from tqdm import tqdm

def clean_mammogram_absolute(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # 1. Binarização simples para separar o fundo
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

    # 2. Encontra Componentes Conectados (melhor que contornos simples para separar ruídos soltos)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Se só achou o fundo, retorna a imagem (algo deu muito errado)
    if num_labels <= 1: return img

    # 3. Ignora o fundo (label 0) e acha a maior "bolha" de pixels (a mama)
    # stats[:, cv2.CC_STAT_AREA] contém a área de cada componente
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1 # +1 porque ignoramos o fundo (índice 0)

    # 4. Cria uma máscara onde APENAS a maior bolha existe
    mask = np.zeros_like(img)
    mask[labels == largest_label] = 255

    # 5. Fechamento Morfológico: tapa pequenos buracos pretos que possam ter ficado dentro da mama
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. Aplica a máscara e apaga tudo ao redor
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 7. Corta a imagem (Bounding Box) na linha exata da mama, tirando o excesso de fundo
    x, y, w, h = cv2.boundingRect(mask)
    margin = 5
    H, W = img.shape
    x_min, y_min = max(0, x - margin), max(0, y - margin)
    x_max, y_max = min(W, x + w + margin), min(H, y + h + margin)

    final_cropped = masked_img[y_min:y_max, x_min:x_max]
    
    return final_cropped


# Configuração de caminhos (ajuste se necessário)
base_path = "/home/pedro-furlan/Área de trabalho/codigos"
input_root = os.path.join(base_path, "dataset/dataset_train") # Ou a pasta original com as letras
output_root = os.path.join(base_path, "dataset_limpo/dataset_train")

# Loop recursivo
for root, dirs, files in os.walk(input_root):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: continue

    relative_path = os.path.relpath(root, input_root)
    current_output_dir = os.path.join(output_root, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for filename in tqdm(image_files):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(current_output_dir, filename)
        
        cleaned = clean_mammogram_absolute(in_path)
        if cleaned is not None:
            cv2.imwrite(out_path, cleaned)