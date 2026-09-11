import cv2
import os
import numpy as np
from tqdm import tqdm

def standardize_orientation(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    
    M = cv2.moments(thresh)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        W = img.shape[1]
        
        if cx > W / 2:
            img = cv2.flip(img, 1)
            
    cv2.imwrite(output_path, img)

def process_dataset(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files: continue
        
        rel_path = os.path.relpath(root, input_folder)
        current_output_dir = os.path.join(output_folder, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        for f in tqdm(image_files, desc=f"Processando {rel_path}"):
            in_path = os.path.join(root, f)
            out_path = os.path.join(current_output_dir, f)
            standardize_orientation(in_path, out_path)

base_path = "/home/pedro-furlan/Área de trabalho/codigos"

input_train = os.path.join(base_path, "dataset_limpo/dataset_train")
output_train = os.path.join(base_path, "dataset_orientado/dataset_train")
process_dataset(input_train, output_train)

input_val = os.path.join(base_path, "dataset_limpo/dataset_val")
output_val = os.path.join(base_path, "dataset_orientado/dataset_val")
process_dataset(input_val, output_val)