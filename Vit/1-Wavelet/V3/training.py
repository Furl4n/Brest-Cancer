from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import os
from transformers.trainer_utils import get_last_checkpoint
import pywt
import cv2
import numpy as np
from PIL import Image

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

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    WaveletMosaicTransform(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2460, 0.2460, 0.2460], std=[0.2054, 0.2054, 0.2054])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    WaveletMosaicTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2460, 0.2460, 0.2460], std=[0.2054, 0.2054, 0.2054])
])

# Carregue o dataset
train_ds = ImageFolder(root="dataset_limpo/dataset_train", transform=transform_train)
val_ds = ImageFolder(root="dataset_limpo/dataset_val", transform=transform_val)

# Modelo ViT com 2 classes - configura o nome das duas classes
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,
    ignore_mismatched_sizes=True,
    id2label={0: "normal", 1: "cancer", 2: "benigno"},
    label2id={"normal": 0, "cancer": 1, "benigno": 2}
)

#Agrupa as imagens em labels
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}

#configura os parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="Vit/Vit_checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    eval_strategy="steps",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir='./logs/Treino-wavelet-V7',
    save_total_limit=2,
    eval_steps=400,
    save_steps=400,
    fp16=True,
    dataloader_num_workers=2,
    warmup_ratio=0.10,
    learning_rate=1e-5,
    weight_decay=0.05,
    load_best_model_at_end=True,      
    metric_for_best_model="eval_f1", # ou "eval_f1", "eval_accuracy".arm
    greater_is_better=True          # False para loss, True se usar accuracy/f1
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro')
    rec = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

#Early stop
early_stop = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.0005
)

#cria o objeto que gerencia o os cliclos de treino e avaliação
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)

last_checkpoint = get_last_checkpoint(training_args.output_dir) #se for apenas continuar de um checkpoint

print("começo do treino\n")

if last_checkpoint is not None:
    print(f"Retomando o treinamento do checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Nenhum checkpoint encontrado. Iniciando do zero.")
    trainer.train()

metrics = trainer.evaluate()
print(metrics)