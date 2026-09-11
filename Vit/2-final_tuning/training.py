from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from transformers.trainer_utils import get_last_checkpoint

transform_train = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2460, 0.2460, 0.2460], std=[0.2054, 0.2054, 0.2054])
])

transform_val = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2460, 0.2460, 0.2460], std=[0.2054, 0.2054, 0.2054])
])

# Carregue o dataset
train_ds = ImageFolder(root="dataset_limpo/dataset_train", transform=transform_train)
val_ds = ImageFolder(root="dataset_limpo/dataset_val", transform=transform_val)

model = ViTForImageClassification.from_pretrained(
    "/home/pedro-furlan/Área de trabalho/checkpoint-4400",
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
    logging_dir='./logs/Treino-final_limpo',
    save_total_limit=2,
    eval_steps=400,
    save_steps=400,
    fp16=True,
    dataloader_num_workers=2,
    warmup_ratio=0.10,
    learning_rate=5e-6,
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