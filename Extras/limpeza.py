import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_background_and_text(image_path):
    # Carrega a imagem original
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # 1. Aplica um desfoque leve para reduzir ruído
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Binarização simples para separar o que é "coisa" do que é fundo preto
    # Usamos um threshold baixo (ex: 15) para pegar até as partes escuras da mama
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    # 3. Encontra os contornos na imagem binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # 4. Assume que o maior contorno da imagem é a mama
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Cria uma máscara preta do mesmo tamanho da imagem
    mask = np.zeros_like(img)

    # 6. Desenha o maior contorno na máscara com a cor branca (preenchido)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 7. Aplica a máscara na imagem original
    # Tudo o que estiver fora do contorno da mama vira preto absoluto (0)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

# Configuração de caminhos
base_path = "/home/pedro-furlan/Área de trabalho/dataset"
input_root = os.path.join("dataset/dataset_train")
output_root = os.path.join("dataset_limpo/dataset_train")

# Loop recursivo (o mesmo do script anterior)
for root, dirs, files in os.walk(input_root):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: continue

    relative_path = os.path.relpath(root, input_root)
    current_output_dir = os.path.join(output_root, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for filename in tqdm(image_files):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(current_output_dir, filename)
        
        cleaned = remove_background_and_text(in_path)
        if cleaned is not None:
            cv2.imwrite(out_path, cleaned)