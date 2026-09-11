import cv2
import numpy as np
import os
from tqdm import tqdm

def clean_mammogram_area_only(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    H, W = img.shape
    img_work = img.copy()
    
    img_work[0:20, :] = 0
    img_work[-20:, :] = 0
    img_work[:, 0:20] = 0
    img_work[:, -20:] = 0

    _, thresh = cv2.threshold(img_work, 15, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    if num_labels <= 1: return img

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    mask = np.zeros_like(img)
    mask[labels == largest_label] = 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.dilate(mask, dilate_kernel)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(mask)
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(W, x + w), min(H, y + h)

    return masked_img[y_min:y_max, x_min:x_max]

base_path = "/home/pedro-furlan/Área de trabalho/codigos"
input_root = os.path.join(base_path, "dataset/dataset_val") # Ou a pasta original com as letras
output_root = os.path.join(base_path, "dataset_limpo/dataset_val")

for root, dirs, files in os.walk(input_root):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: continue

    relative_path = os.path.relpath(root, input_root)
    current_output_dir = os.path.join(output_root, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for filename in tqdm(image_files):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(current_output_dir, filename)
        
        cleaned = clean_mammogram_area_only(in_path)
        if cleaned is not None:
            cv2.imwrite(out_path, cleaned)