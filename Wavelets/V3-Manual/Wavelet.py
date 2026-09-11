import pywt
import cv2
import numpy as np

#Transforma em um vetor e passa para float
img_original = (cv2.imread('dataset/dataset_val/Benign/0029/C_0029_1.LEFT_MLO.jpg', cv2.IMREAD_GRAYSCALE)).astype(np.float32)

#Faz a decomposição
coeffs = pywt.wavedec2(img_original, 'db4', 'symmetric', level=2)

#restaura a foto normal - usando para comparar qualidade
img_restaurada = pywt.waverec2(coeffs, 'db4', 'symmetric')

#Salva em png
cv2.imwrite('Wavelets/V3-manual/Resultados/Original.png', img_original)
cv2.imwrite('Wavelets/V3-manual/Resultados/Restaurada.png', img_restaurada)


# Função para normalizar o coeficiente para 0-255 e uint8 (peguei pronto)
def norm_and_save(arr, fname):
    arr_norm = arr - np.min(arr)
    if np.max(arr_norm) > 0:
        arr_norm = arr_norm / np.max(arr_norm)
    img_uint8 = (arr_norm * 255).astype(np.uint8)

    h, w = img_uint8.shape[:2]
    scale = 1080/h
    new_width = int(w * scale)
    img_final = cv2.resize(img_uint8, (new_width, 1080))

    cv2.imwrite(fname, img_final)
    return img_final

#Salva o LL

# Salva cada coeficiente de cada nível
for level, coefs in enumerate(coeffs[1:], 1):
    LH, HL, HH = coefs
    img_LH = norm_and_save(LH, f"Wavelets/V3-Manual/Resultados/LH{level}.png")
    img_HL = norm_and_save(HL, f"Wavelets/V3-Manual/Resultados/HL{level}.png")
    img_HH = norm_and_save(HH, f"Wavelets/V3-Manual/Resultados/HH{level}.png")
    img_LL = norm_and_save(coeffs[0], "Wavelets/V3-Manual/Resultados/LL.png")
    img_LL = cv2.resize(img_LL, (img_HH.shape[1], img_HH.shape[0]))

    print('Shapes:', img_LL.shape, img_LH.shape, img_HL.shape, img_HH.shape)
    print('Types:', img_LL.dtype, img_LH.dtype, img_HL.dtype, img_HH.dtype)


    top_row = cv2.hconcat([img_LL, img_LH])
    bottom_row = cv2.hconcat([img_HL, img_HH])
    block = cv2.vconcat([top_row, bottom_row]);
    cv2.imwrite( f'Wavelets/V3-Manual/Resultados/BLOCK{level}.png', block)



# MSE analisa a diferença entre as imagens -Melhor
print("MSE (Mean Squared Error): ", np.mean((img_original - img_restaurada) ** 2))

# SNR analisa o ruido na imagem +melhor
signal_power = np.mean(img_original ** 2)
noise_power = np.mean((img_original - img_restaurada) ** 2)
print("SNR (Signal-to-Noise Ratio):", 10 * np.log10(signal_power / noise_power), "db")
