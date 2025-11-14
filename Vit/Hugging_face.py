from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

image = Image.open("mama.pgm")
image_rgb = image.convert("RGB")

# Carregar modelo pré-treinado
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Processar e classificar imagem
inputs = processor(images=image_rgb, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Mostrar a classe prevista
print("Classe prevista:", model.config.id2label[predicted_class_idx])