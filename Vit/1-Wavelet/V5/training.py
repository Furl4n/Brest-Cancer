from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

MEAN = [0.3639675974845886, 0.49822163581848145, 0.4940685033798218]
STD = [0.3144570589065552, 0.017851922661066055, 0.02476944774389267]

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_ds = ImageFolder(
    root="dataset_wavelet/dataset_train",
    transform=transform_train
)

val_ds = ImageFolder(
    root="dataset_wavelet/dataset_val",
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

model_stage1 = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=3,
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id
)

freeze_backbone_keep_head(model_stage1)

training_args_stage1 = TrainingArguments(
    output_dir="Vit/Vit_wavelet_3ch_stage1",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir="./logs/Treino-wavelet-3ch-stage1",
    save_total_limit=2,
    fp16=True,
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

print("===== FASE 1: TREINANDO SÓ A CABEÇA =====\n")
trainer_stage1.train()

metrics_stage1 = trainer_stage1.evaluate()
print("Métricas Fase 1:")
print(metrics_stage1)

model_stage2 = model_stage1
unfreeze_all(model_stage2)

training_args_stage2 = TrainingArguments(
    output_dir="Vit/Vit_wavelet_3ch_stage2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=40,
    report_to="tensorboard",
    logging_dir="./logs/Treino-wavelet-3ch-stage2",
    save_total_limit=2,
    fp16=True,
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

print("\n===== FASE 2: FINE-TUNING COMPLETO =====\n")
trainer_stage2.train()

metrics_stage2 = trainer_stage2.evaluate()
print("Métricas Fase 2:")
print(metrics_stage2)