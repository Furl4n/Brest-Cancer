from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from transformers.trainer_utils import get_last_checkpoint
import pywt
import numpy as np
from PIL import Image

class WaveletMosaicTransform:
    def __init__(self, wavelet='db4'):
        self.wavelet = wavelet

    def __call__(self, img):
        arr = np.array(img.convert("L"), dtype=np.float32)
        
        coeffs2 = pywt.dwt2(arr, self.wavelet, mode='symmetric')
        LL, (LH, HL, HH) = coeffs2
        
        def norm(c):
            c_min, c_max = np.percentile(c, 1), np.percentile(c, 99)
            c = np.clip(c, c_min, c_max)

            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = np.zeros_like(c)

            return c * 255.0
            
        top = np.hstack((norm(LL), norm(HL)))
        bottom = np.hstack((norm(LH), norm(HH)))
        mosaic = np.vstack((top, bottom))
        
        return Image.fromarray(mosaic.astype(np.uint8), mode='L')

transform_train = transforms.Compose([
    WaveletMosaicTransform(wavelet='db4'),
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.RandomRotation(5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4636, 0.4636, 0.4636], std=[0.1667, 0.1667, 0.1667])
])

transform_val = transforms.Compose([
    WaveletMosaicTransform(wavelet='db4'),
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4636, 0.4636, 0.4636], std=[0.1667, 0.1667, 0.1667])
])

train_ds = ImageFolder(root="dataset_orientado/dataset_train", transform=transform_train)
val_ds = ImageFolder(root="dataset_orientado/dataset_val", transform=transform_val)

model = ViTForImageClassification.from_pretrained(
    "/home/pedro-furlan/Área de trabalho/checkpoint-4400",
    num_labels=3,
    ignore_mismatched_sizes=True,
    id2label={0: "normal", 1: "cancer", 2: "benigno"},
    label2id={"normal": 0, "cancer": 1, "benigno": 2}
)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}

training_args = TrainingArguments(
    output_dir="Vit/Vit_checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir='./logs/Treino-wavelet-V7-final',
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,
    warmup_ratio=0.10,
    learning_rate=5e-6,
    weight_decay=0.05,
    load_best_model_at_end=True,      
    metric_for_best_model="eval_f1",
    greater_is_better=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro', zero_division=0)
    rec = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

early_stop = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.0005
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)

last_checkpoint = get_last_checkpoint(training_args.output_dir)

print("Começo do treino final\n")

if last_checkpoint is not None:
    print(f"Retomando o treinamento do checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Nenhum checkpoint encontrado na pasta. Iniciando transfer learning...")
    trainer.train()

metrics = trainer.evaluate()
print(metrics)