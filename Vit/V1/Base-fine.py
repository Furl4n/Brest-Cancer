from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

# Preprocessamento igual ao ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2048, 0.2048, 0.2048], std=[0.2048, 0.2048, 0.2048])
])

# Carregue o dataset
dataset = ImageFolder(root="/home/pedro-furlan/Área de trabalho/codigos/dataset", transform=transform)

# Divida entre treino e validação - 80% treinamento e 20% validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

# Modelo ViT com 2 classes - configura o nome das duas classes
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.config.num_labels = 2
model.config.id2label = {0: "nao_cancer", 1: "cancer"}
model.config.label2id = {"nao_cancer": 0, "cancer": 1}


#Agrupa as imagens em labels
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": images, "labels": labels}

#configura os parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="/home/pedro-furlan/Área de trabalho/codigos/Vit/vit_cancer_finetune",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    eval_strategy="steps",
    logging_steps=20,
    save_total_limit=2,
    eval_steps=200,                                  
    save_steps=200,
    fp16=True,
    dataloader_num_workers=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='binary')
    rec = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

#cria o objeto que gerencia o os cliclos de treino e avaliação
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

print("começo do treino\n")

#inicia o treinamento
trainer.train()

metrics = trainer.evaluate()
print(metrics)