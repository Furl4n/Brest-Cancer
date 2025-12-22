import os
import pywt
import cv2
import numpy as np

# Pastas
INPUT_DIR  = "dataset"                 # raiz do dataset .jpg
OUTPUT_DIR = "Wavelet-dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def norm_and_save(arr, fname):
    arr_norm = arr - np.min(arr)
    if np.max(arr_norm) > 0:
        arr_norm = arr_norm / np.max(arr_norm)
    img_uint8 = (arr_norm * 255).astype(np.uint8)

    h, w = img_uint8.shape[:2]
    scale = 1080 / h
    new_width = int(w * scale)
    img_final = cv2.resize(img_uint8, (new_width, 1080))

    cv2.imwrite(fname, img_final)
    return img_final

# Percorre todo o dataset
for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.lower().endswith(".jpg"):
            continue

        in_path = os.path.join(root, fname)

        # Caminho relativo para espelhar estrutura
        rel_dir = os.path.relpath(root, INPUT_DIR)
        out_base_dir = os.path.join(OUTPUT_DIR, rel_dir, os.path.splitext(fname)[0])
        os.makedirs(out_base_dir, exist_ok=True)

        # 1) lê em grayscale e float32
        img_original = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        img_original = img_original.astype(np.float32)

        # 2) decomposição wavelet
        coeffs = pywt.wavedec2(img_original, 'db4', 'symmetric', level=2)

        def norm_and_resize(arr, target_h=1080):
            arr_norm = arr - np.min(arr)
            if np.max(arr_norm) > 0:
                arr_norm = arr_norm / np.max(arr_norm)
            img_uint8 = (arr_norm * 255).astype(np.uint8)

            h, w = img_uint8.shape[:2]
            scale = 1080/h
            new_width = int(w * scale)
            img_final = cv2.resize(img_uint8, (new_width, 1080))

            return img_final

        # 3) salva LL, LH, HL, HH e blocos por nível
        for level, coefs in enumerate(coeffs[1:], 1):
            LH, HL, HH = coefs
            img_LH = norm_and_resize(LH)
            img_HL = norm_and_resize(HL)
            img_HH = norm_and_resize(HH)
            img_LL = norm_and_resize(coeffs[0])
            img_LL = cv2.resize(img_LL, (img_HH.shape[1], img_HH.shape[0]))

            top_row = cv2.hconcat([img_LL, img_LH])
            bottom_row = cv2.hconcat([img_HL, img_HH])
            block = cv2.vconcat([top_row, bottom_row])
            cv2.imwrite(os.path.join(out_base_dir, f"BLOCK{level}.png"), block)

        print("Processado:", in_path, "->", out_base_dir)
