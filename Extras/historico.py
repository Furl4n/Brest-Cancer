import torch
from transformers import ViTConfig, ViTForImageClassification

# Substitua pelo caminho da pasta do seu checkpoint específico
caminho_do_checkpoint = "/home/pedro-furlan/Área de trabalho/checkpoint-4400"
caminho_do_arquivo = f"{caminho_do_checkpoint}/training_args.bin"

# Carrega o arquivo binário usando o PyTorch
try:
    args = torch.load(caminho_do_arquivo, map_location=torch.device('cpu'), weights_only=False)
    
    # Imprime os argumentos mais cruciais para o seu experimento
    print(f"--- Configurações Recuperadas do Checkpoint ---")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Warmup Ratio: {args.warmup_ratio}")
    print(f"Num Epochs: {args.num_train_epochs}")
    print(f"Batch Size (Train): {args.per_device_train_batch_size}")
    print(f"Scheduler Type: {args.lr_scheduler_type}")
    print(f"Optimizer: {args.optim}")
    print(f"Pasta Original de Saída: {args.output_dir}")
    
    # Se quiser ver a lista gigante com ABSOLUTAMENTE TODOS os parâmetros, descomente a linha abaixo:
    # print(args)

except FileNotFoundError:
    print(f"Erro: O arquivo training_args.bin não foi encontrado na pasta {caminho_do_checkpoint}")


print("\n[2] ARQUITETURA DO MODELO (Canais da Imagem):")

try:
    # Checando o arquivo de texto (config.json)
    config = ViTConfig.from_pretrained(caminho_do_checkpoint)
    print(f" -> Canais registrados no config.json: {config.num_channels}")
    
    # Checando a matemática física (A prova real)
    print(" -> Carregando os pesos físicos da rede (isso pode levar alguns segundos)...")
    model = ViTForImageClassification.from_pretrained(
        caminho_do_checkpoint, 
        ignore_mismatched_sizes=True, 
        device_map="cpu" # <--- Garante que a análise não vai derrubar um treino ativo na GPU!
    )
    
    # Extraindo o formato do filtro de entrada
    formato_pesos = model.vit.embeddings.patch_embeddings.projection.weight.shape
    canais_reais = formato_pesos[1]
    
    print(f" -> Canais esculpidos fisicamente nos neurônios: {canais_reais}")
    
    print("\n--- DIAGNÓSTICO FINAL ---")
    if canais_reais == 3:
        print("📋 RESULTADO: Imagem Normal (RGB/Grayscale).")
        print("   Este modelo é de FINE-TUNING BRUTO (Estágio 2).")
    elif canais_reais == 4:
        print("🌊 RESULTADO: Mosaico Wavelet (4 Canais).")
        print("   Este modelo é de APRENDIZADO DE FREQUÊNCIA (Estágio 1).")
    else:
        print(f"⚠️ RESULTADO: Formato incomum. O modelo espera {canais_reais} canais.")

except Exception as e:
    print(f" -> ERRO ao analisar a arquitetura do modelo: {e}")