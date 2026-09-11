import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_artifacts_and_crop(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

    masked_img = cv2.bitwise_and(img, img, mask=mask_dilated)

    x, y, w, h = cv2.boundingRect(mask_dilated)
    
    margin = 10
    H, W = img.shape
    x_min, y_min = max(0, x - margin), max(0, y - margin)
    x_max, y_max = min(W, x + w + margin), min(H, y + h + margin)

    final_cropped = masked_img[y_min:y_max, x_min:x_max]

    return final_cropped

base_path = "/home/pedro-furlan/Área de trabalho/dataset"
input_root = os.path.join("dataset/dataset_train")
output_root = os.path.join("dataset_limpo/dataset_train")

for root, dirs, files in os.walk(input_root):
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: continue

    relative_path = os.path.relpath(root, input_root)
    current_output_dir = os.path.join(output_root, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for filename in tqdm(image_files):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(current_output_dir, filename)
        
        cleaned = remove_artifacts_and_crop(in_path)
        if cleaned is not None:
            cv2.imwrite(out_path, cleaned)