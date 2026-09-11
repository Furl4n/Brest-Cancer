import json
from torch.utils.tensorboard import SummaryWriter
import os

pasta_checkpoint = "Vit/vit_cancer_finetune/checkpoint-6400"
nome_experimento = "Basico-V2" # Mude isso para identificar no gráfico

# Caminho do arquivo que tem a história do treino
arquivo_estado = f"{pasta_checkpoint}/trainer_state.json"
pasta_logs = f"logs/{nome_experimento}" # Onde o TensorBoard vai ler

print(f"Lendo o passado de: {arquivo_estado}...")

try:
    with open(arquivo_estado, 'r') as f:
        dados = json.load(f)
        historico = dados.get("log_history", [])

    writer_treino = SummaryWriter(log_dir=os.path.join(pasta_logs, "treino"))
    writer_val = SummaryWriter(log_dir=os.path.join(pasta_logs, "validacao"))

    # Percorre cada passo salvo no seu json antigo
    for log in historico:
        passo = log.get("step", 0)
        
        if "loss" in log:
            writer_treino.add_scalar("Treino/Loss", log["loss"], passo)
            writer_treino.add_scalar("Comparação/Loss", log["loss"], passo)
            writer_treino.add_scalar("Treino/Learning_Rate", log["learning_rate"], passo)
            
        if "eval_loss" in log:    
            writer_val.add_scalar("Validacao/Loss", log["eval_loss"], passo)
            writer_val.add_scalar("Validacao/F1_Score", log.get("eval_f1", 0), passo)
            writer_val.add_scalar("Comparação/Loss", log["eval_loss"], passo)
            writer_val.add_scalar("Validacao/Acuracia", log.get("eval_accuracy", 0), passo)

    # Fecha e salva os gráficos
    writer_treino.close()
    writer_val.close()
    print(f"Sucesso! Gráficos gerados e salvos na pasta: {pasta_logs}")
    print("Para ver, digite no terminal: tensorboard --logdir ./logs")

except FileNotFoundError:
    print(f"Erro: Não achei o arquivo trainer_state.json na pasta {pasta_checkpoint}")