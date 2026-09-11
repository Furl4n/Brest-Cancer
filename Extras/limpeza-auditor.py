import cv2
import numpy as np
import os
from tqdm import tqdm

def detect_artifact(clean_img_path):
    img = cv2.imread(clean_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    _, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            return True
    return False

def extreme_clean(original_path):
    img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    H, W = img.shape
    img_work = img.copy()
    
    img_work[0:40, :] = 0
    img_work[-40:, :] = 0
    img_work[:, 0:40] = 0
    img_work[:, -40:] = 0
    
    _, thresh = cv2.threshold(img_work, 15, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 65))
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
clean_root = os.path.join(base_path, "dataset_limpo/dataset_train")
dirty_root = os.path.join(base_path, "dataset/dataset_train")

for root, _, files in os.walk(clean_root):
    for f in tqdm(files):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            clean_path = os.path.join(root, f)
            
            if detect_artifact(clean_path):
                rel_path = os.path.relpath(clean_path, clean_root)
                dirty_path = os.path.join(dirty_root, rel_path)
                
                if os.path.exists(dirty_path):
                    fixed_img = extreme_clean(dirty_path)
                    if fixed_img is not None:
                        cv2.imwrite(clean_path, fixed_img)
                        tqdm.write(f"[CORRIGIDO] {rel_path}")