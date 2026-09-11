import os
from pathlib import Path
import numpy as np
import pywt
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIGURAÇÃO DE CAMINHOS
# =========================
SOURCE_ROOT = Path("dataset_orientado")
TARGET_ROOT = Path("dataset_wavelet")

SPLITS = ["dataset_train", "dataset_val"]
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

WAVELET = "db4"
OUTPUT_SIZE = (224, 224)

def is_image_file(path: Path):
    return path.suffix.lower() in VALID_EXTS

def norm_band(band):
    band = band.astype(np.float32)
    b_min, b_max = np.percentile(band, 1), np.percentile(band, 99)
    band = np.clip(band, b_min, b_max)

    if b_max > b_min:
        band = (band - b_min) / (b_max - b_min)
    else:
        band = np.zeros_like(band, dtype=np.float32)

    band = (band * 255.0).clip(0, 255).astype(np.uint8)
    return band

def resize_band_to_224(band_uint8):
    pil_img = Image.fromarray(band_uint8, mode="L")
    pil_img = pil_img.resize(OUTPUT_SIZE, Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)

def build_wavelet_rgb_image(img, wavelet="db4"):
    gray = np.array(img.convert("L"), dtype=np.float32)

    coeffs2 = pywt.dwt2(gray, wavelet, mode="symmetric")
    LL, (LH, HL, HH) = coeffs2

    ll_img = resize_band_to_224(norm_band(LL))
    lh_img = resize_band_to_224(norm_band(LH))
    hl_img = resize_band_to_224(norm_band(HL))

    rgb = np.stack([ll_img, lh_img, hl_img], axis=-1)
    return Image.fromarray(rgb, mode="RGB")

def process_and_save_image(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        wavelet_rgb = build_wavelet_rgb_image(img, wavelet=WAVELET)
        wavelet_rgb.save(dst_path.with_suffix(".png"))

def build_wavelet_dataset():
    total = 0

    for split in SPLITS:
        split_dir = SOURCE_ROOT / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                total += sum(
                    1 for f in class_dir.rglob("*")
                    if f.is_file() and is_image_file(f)
                )

    pbar = tqdm(total=total, desc="Gerando dataset wavelet 3 canais")

    for split in SPLITS:
        split_dir = SOURCE_ROOT / split
        target_split_dir = TARGET_ROOT / split

        if not split_dir.exists():
            print(f"[AVISO] Split não encontrado: {split_dir}")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            target_class_dir = target_split_dir / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for src_file in class_dir.rglob("*"):
                if not src_file.is_file() or not is_image_file(src_file):
                    continue

                rel_path = src_file.relative_to(class_dir)
                dst_file = target_class_dir / rel_path

                try:
                    process_and_save_image(src_file, dst_file)
                except Exception as e:
                    print(f"[ERRO] Falha em {src_file}: {e}")

                pbar.update(1)

    pbar.close()
    print(f"\nDataset wavelet 3 canais criado em: {TARGET_ROOT}")

if __name__ == "__main__":
    build_wavelet_dataset()