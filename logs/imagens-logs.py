import torch
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor
import os

# Lista de experimentos que você quer atualizar
# Formato: (Nome_no_TensorBoard, Caminho_da_Imagem, Passo_Final)
experimentos = [
    ("Exp_V1_Base", "Vit/1-base/confusion_matrix.png", 1200),
    ("Exp_V2_Wavelet", "Vit/2-wavelet/confusion_matrix.png", 4500),
    ("Exp_V3_Final", "Vit/3-final/confusion_matrix.png", 8000)
]

for nome, path_img, step in experimentos:
    if os.path.exists(path_img):
        # Iclui o log na pasta de validação de cada experimento
        log_dir = f"./logs/{nome}/validacao"
        writer = SummaryWriter(log_dir=log_dir)
        
        img = PIL.Image.open(path_img)
        img_tensor = ToTensor()(img)
        
        # O nome 'Matriz_Confusao_Final' será o mesmo para todos
        # permitindo comparar as imagens no TensorBoard
        writer.add_image('Matriz_Confusao_Final', img_tensor, step)
        writer.close()
        print(f"✅ Sucesso: {nome}")
    else:
        print(f"⚠️ Ignorado: {path_img} não encontrado.")