import pywt
import cv2
import numpy as np

# --- 1. Carregamento e Decomposição ---
caminho_imagem = '/home/pedro-furlan/Área de trabalho/IC - Breast cancer/Codigos/Wavelets/mama.pgm'

# Carrega a imagem em escala de cinza e converte para float
img_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
if img_original is None:
    print(f"ERRO: Não foi possível carregar a imagem em '{caminho_imagem}'")
    exit()
img_original = img_original.astype(np.float32)

# Aplica a Transformada Wavelet 2D com 2 níveis de decomposição
coeffs = pywt.wavedec2(img_original, 'db4', level=2, mode='symmetric')

# Desempacota os coeficientes
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
print("--- Decomposição Wavelet Realizada ---")

# --- 2. Função para Normalizar e Salvar com Contraste ---
def salvar_coeficiente_com_contraste(nome_arquivo, coeficiente_matrix):
    """Normaliza coeficientes e aplica realce forte de contraste (tipo mamografia)."""
    
    # Evita divisão por zero
    max_val = np.max(np.abs(coeficiente_matrix))
    if max_val < 1e-8:
        print(f"Aviso: coeficientes muito baixos em {nome_arquivo}.")
        img_final = np.zeros_like(coeficiente_matrix, dtype=np.uint8)
    else:
        # Normaliza preservando o sinal
        coef_norm = coeficiente_matrix / max_val
        
        # Mapeia -1..1 para 0..255
        img_bw = ((coef_norm + 1) / 2) * 255
        img_bw = np.clip(img_bw, 0, 255).astype(np.uint8)
        
        # Aplica CLAHE (realce de contraste local)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_final = clahe.apply(img_bw)

    cv2.imwrite(nome_arquivo, img_final)
    print(f"Arquivo '{nome_arquivo}' salvo com contraste preto/branco aprimorado.")


# --- 3. Salvar os Coeficientes como Imagens Visíveis ---
print("\n--- Salvando os coeficientes como imagens ---")

# Nível 2 (mais geral, menos detalhes)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Aproximacao_LL2.png', cA2)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Horizontal_LH2.png', cH2)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Vertical_HL2.png', cV2)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Diagonal_HH2.png', cD2)

# Nível 1 (detalhes finos)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Horizontal_LH1.png', cH1)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Vertical_HL1.png', cV1)
salvar_coeficiente_com_contraste('Wavelets/V1/Resultados/resultado_Detalhe_Diagonal_HH1.png', cD1)

print("\n--- Avaliação de reconstrução ---")

# --- 4. Reconstrução da Imagem ---
img_restaurada = pywt.waverec2(coeffs, 'db4', mode='symmetric')
img_restaurada = np.clip(img_restaurada, 0, 255)

# --- 5. Métricas de qualidade ---
# MSE
mse = np.mean((img_original - img_restaurada) ** 2)
print("MSE (Mean Squared Error):", mse)

# SNR
signal_power = np.mean(img_original ** 2)
noise_power = mse
snr = 10 * np.log10(signal_power / noise_power)
print("SNR (Signal-to-Noise Ratio):", snr, "dB")

print("\nProcesso concluído com sucesso.")
