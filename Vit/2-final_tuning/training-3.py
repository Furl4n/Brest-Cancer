from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch


# =========================================================
# ESTATÍSTICAS REAIS DO DATASET NORMAL
# =========================================================
MEAN = [0.2460, 0.2460, 0.2460]
STD = [0.2180, 0.2180, 0.2180]


# =========================================================
# CAMINHOS
# =========================================================
TRAIN_ROOT = "dataset_orientado/dataset_train"
VAL_ROOT = "dataset_orientado/dataset_val"

# coloque aqui o melhor checkpoint do treino wavelet
WAVELET_CHECKPOINT = "/home/pedro-furlan/Área de trabalho/checkpoint-1552"


# =========================================================
# TRANSFORMS
# Se as imagens normais forem grayscale, isso garante 3 canais.
# =========================================================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# =========================================================
# DATASETS
# =========================================================
train_ds = ImageFolder(
    root=TRAIN_ROOT,
    transform=transform_train
)

val_ds = ImageFolder(
    root=VAL_ROOT,
    transform=transform_val
)

id2label = {0: "normal", 1: "cancer", 2: "benigno"}
label2id = {"normal": 0, "cancer": 1, "benigno": 2}


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0)
    }


def freeze_backbone_keep_head(model):
    for param in model.vit.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# =========================================================
# CARREGA O MODELO JÁ TREINADO NO WAVELET
# =========================================================
model_stage1 = ViTForImageClassification.from_pretrained(
    WAVELET_CHECKPOINT,
    num_labels=len(train_ds.classes),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id
)


# =========================================================
# FASE 1: TREINA SÓ A CABEÇA SOBRE AS IMAGENS NORMAIS
# =========================================================
freeze_backbone_keep_head(model_stage1)


training_args_stage1 = TrainingArguments(
    output_dir="Vit/Vit_final_stage1",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir="./logs/Treino-final-stage1",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    warmup_ratio=0.05,
    learning_rate=3e-6,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    remove_unused_columns=False,
    seed=42
)


early_stop_stage1 = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0005
)


trainer_stage1 = Trainer(
    model=model_stage1,
    args=training_args_stage1,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[early_stop_stage1]
)


print("===== FASE 1: REAPROVEITANDO O WAVELET E TREINANDO A CABEÇA =====\n")
trainer_stage1.train()


metrics_stage1 = trainer_stage1.evaluate()
print("Métricas Fase 1:")
print(metrics_stage1)


# =========================================================
# FASE 2: DESCONGELA TUDO E SEGUE O FINE-TUNING
# =========================================================
model_stage2 = model_stage1
unfreeze_all(model_stage2)


training_args_stage2 = TrainingArguments(
    output_dir="Vit/Vit_final_stage2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir="./logs/Treino-final-stage2",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    warmup_ratio=0.05,
    learning_rate=2e-6,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    remove_unused_columns=False,
    seed=42
)


early_stop_stage2 = EarlyStoppingCallback(
    early_stopping_patience=4,
    early_stopping_threshold=0.0005
)


trainer_stage2 = Trainer(
    model=model_stage2,
    args=training_args_stage2,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[early_stop_stage2]
)


print("\n===== FASE 2: FINE-TUNING COMPLETO EM IMAGENS NORMAIS =====\n")
trainer_stage2.train()


metrics_stage2 = trainer_stage2.evaluate()
print("Métricas Fase 2:")
print(metrics_stage2)

print("\nMelhor checkpoint:", trainer_stage2.state.best_model_checkpoint)
print("Melhor métrica:", trainer_stage2.state.best_metric)