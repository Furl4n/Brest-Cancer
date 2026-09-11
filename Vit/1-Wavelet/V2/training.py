from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.nn as nn
import pywt
import numpy as np
from PIL import Image
import cv2

class WaveletTransform:
    def __init__(self):  #construtor da classe - self=objeto
        self.img_size = 224
        self.wavelet = 'db4'
        self.levels = 1
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, pil_img):
        # PIL é a imagem original - transforma em Grayscale e float
        img_np = np.array(pil_img.convert('L')).astype(np.float32)
        
        #Wavelet
        coeffs = pywt.wavedec2(img_np, self.wavelet, mode='symmetric', level=self.levels)
        
        #salva as decomposições - pega o último nível
        LL = coeffs[0]
        LH, HL, HH = coeffs[1]
        
        # normaliza para 224x224
        def normalize_tensor(band):
            band_pil = Image.fromarray(((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8))
            band_resized = self.resize(band_pil)
            return transforms.ToTensor()(band_resized)
        
        bands = [
            normalize_tensor(LL),
            normalize_tensor(LH),  
            normalize_tensor(HL),
            normalize_tensor(HH)
        ]
        
        return torch.cat(bands, dim=0)
    
transform = WaveletTransform()

# Carregue o dataset
train_ds = ImageFolder(root="dataset/dataset_train", transform=transform)
val_ds = ImageFolder(root="dataset/dataset_val", transform=transform)

# Modelo ViT com 2 classes - configura o nome das duas classes
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_channels=4, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
model.config.num_labels = 3
model.config.id2label = {0: "normal", 1: "cancer", 2: "benigno"}
model.config.label2id = {"normal": 0, "cancer": 1, "benigno" :2}


#Agrupa as imagens em labels
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}

#configura os parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="Vit/vit_cancer_finetune",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    eval_strategy="steps",
    logging_steps=20,
    save_total_limit=2,
    eval_steps=400,                                  
    save_steps=400,
    fp16=True,
    dataloader_num_workers=2,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
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
    early_stopping_patience=4,
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

print("começo do treino\n")

#inicia o treinamento
trainer.train()

metrics = trainer.evaluate()
print(metrics)