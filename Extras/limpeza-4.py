import cv2
import numpy as np
import os
from tqdm import tqdm

def clean_mammogram_ultimate(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    if num_labels <= 1: return img

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    clean_mask = np.zeros_like(img)
    clean_mask[labels == largest_label] = 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    clean_mask = cv2.dilate(clean_mask, dilate_kernel)

    masked_img = cv2.bitwise_and(img, img, mask=clean_mask)

    x, y, w, h = cv2.boundingRect(clean_mask)
    margin = 5
    H, W = img.shape
    x_min, y_min = max(0, x - margin), max(0, y - margin)
    x_max, y_max = min(W, x + w + margin), min(H, y + h + margin)

    return masked_img[y_min:y_max, x_min:x_max]

base_path = "/home/pedro-furlan/Área de trabalho/codigos"
input_root = os.path.join(base_path, "dataset/dataset_train") # Ou a pasta original com as letras
output_root = os.path.join(base_path, "dataset_limpo/dataset_train")

for root, dirs, files in os.walk(input_root):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: continue

    relative_path = os.path.relpath(root, input_root)
    current_output_dir = os.path.join(output_root, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for filename in tqdm(image_files):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(current_output_dir, filename)
        
        cleaned = clean_mammogram_ultimate(in_path)
        if cleaned is not None:
            cv2.imwrite(out_path, cleaned)