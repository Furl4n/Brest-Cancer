import pywt
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class WaveletMosaicTransform:

    def __init__(self):
        self.wavelet = 'db4'
        self.levels = 1

    def __call__(self, img):
        # 1. Garante que a entrada (PIL Image) seja escala de cinza e numpy array
        img_gray = np.array(img.convert('L'), dtype=np.float32)
        orig_h, orig_w = img_gray.shape
        
        # 2. Decomposição Wavelet Nível 1 (gera exatamente 4 componentes)
        coeffs = pywt.wavedec2(img_gray, self.wavelet, mode='symmetric', level=1)
        LL, (LH, HL, HH) = coeffs
        
        # Função interna rápida para normalizar 0-255
        def normalize(arr):
            arr_min, arr_max = arr.min(), arr.max() #acha o maior e menor valor ao array
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) #Min-Max Scaling: o menor vira 0 e o maior 1, o resto fica entre eles
            return (arr * 255).astype(np.uint8)

        # 3. Normaliza os coeficientes
        LL_norm = normalize(LL)
        LH_norm = normalize(LH)
        HL_norm = normalize(HL)
        HH_norm = normalize(HH)
        
        # Garante que os tamanhos batam exatamente (lidando com pixels ímpares, se houver)
        h, w = LL_norm.shape
        LH_norm = cv2.resize(LH_norm, (w, h))
        HL_norm = cv2.resize(HL_norm, (w, h))
        HH_norm = cv2.resize(HH_norm, (w, h))

        # 4. Constrói o mosaico (LL topo-esq, HL topo-dir, LH base-esq, HH base-dir)
        top_row = np.hstack((LL_norm, HL_norm))
        bottom_row = np.hstack((LH_norm, HH_norm))
        mosaic = np.vstack((top_row, bottom_row))
        
        # 5. Converte para 3 canais RGB (pois o ViT espera 3 canais)
        mosaic_rgb = cv2.merge([mosaic, mosaic, mosaic])

        mosaic_final = cv2.resize(mosaic_rgb, (orig_w, orig_h))
        
        # Retorna como PIL Image para o PyTorch continuar aplicando o ToTensor()
        return Image.fromarray(mosaic_final)
    
wavelet_transform = WaveletMosaicTransform()

test_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    wavelet_transform
])
    
img_teste = test_pipeline(Image.open('dataset/teste/C_0033_1.LEFT_CC.jpg'))

img_teste.save('teste_mosaico_224x224.png')