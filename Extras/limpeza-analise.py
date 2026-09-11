import cv2
import os
import numpy as np
from tqdm import tqdm

def audit_cleaned_dataset(clean_root):
    suspicious_files = []

    for root, _, files in os.walk(clean_root):
        for f in tqdm(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                h, w = img.shape
                
                aspect_ratio = w / float(h)
                
                black_pixels = np.sum(img < 15)
                total_pixels = w * h
                black_ratio = black_pixels / float(total_pixels)

                if aspect_ratio > 1.0 or black_ratio > 0.5:
                    suspicious_files.append({
                        'path': img_path,
                        'aspect_ratio': aspect_ratio,
                        'black_ratio': black_ratio
                    })

    print(f"\nTotal de imagens suspeitas encontradas: {len(suspicious_files)}")
    
    suspicious_files.sort(key=lambda x: x['black_ratio'], reverse=True)

    print("\n--- TOP 30 IMAGENS MAIS SUSPEITAS ---")
    for item in suspicious_files[:30]:
        print(f"[{item['black_ratio']*100:.1f}% Preto | Largura/Altura: {item['aspect_ratio']:.2f}] -> {item['path']}")

clean_root = "dataset_limpo/dataset_val"
audit_cleaned_dataset(clean_root)