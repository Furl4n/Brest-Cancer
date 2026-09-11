import os

def obter_pacientes(caminho_dataset):
    pacientes = set()
    # Entra nas pastas de classes (benigno, cancer, normal)
    for classe in os.listdir(caminho_dataset):
        caminho_classe = os.path.join(caminho_dataset, classe)
        if os.path.isdir(caminho_classe):
            # Entra nas pastas dos pacientes (P_00001, P_00004...)
            for paciente in os.listdir(caminho_classe):
                if os.path.isdir(os.path.join(caminho_classe, paciente)):
                    pacientes.add(paciente)
    return pacientes

# AJUSTE ESTES CAMINHOS PARA AS SUAS PASTAS REAIS
caminho_treino = 'dataset_orientado/dataset_train'
caminho_val = 'dataset_orientado/dataset_val'

pacientes_treino = obter_pacientes(caminho_treino)
pacientes_val = obter_pacientes(caminho_val)

# A prova real: interseção matemática entre os dois conjuntos
vazamento = pacientes_treino.intersection(pacientes_val)

print("=== RELATÓRIO DE AUDITORIA DE DADOS ===")
print(f"Total de Pacientes no Treino: {len(pacientes_treino)}")
print(f"Total de Pacientes na Validação: {len(pacientes_val)}")
print(f"Pacientes em AMBOS (Vazamento): {len(vazamento)}\n")

if len(vazamento) > 0:
    print("❌ ALERTA CRÍTICO: VAZAMENTO DETECTADO!")
    print("Os seguintes pacientes estão corrompendo o teste:")
    print(vazamento)
else:
    print("✅ SUCESSO: Zero vazamento de dados. Os pacientes estão perfeitamente isolados.")