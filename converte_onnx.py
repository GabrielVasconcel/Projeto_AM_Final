import os
import joblib
import numpy as np

# Tenta importar as bibliotecas de conversão
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
    print("Bibliotecas ONNX (skl2onnx, onnxruntime) encontradas. Iniciando conversão.")
except ImportError:
    ONNX_AVAILABLE = False
    print("ERRO: Bibliotecas ONNX não instaladas. Por favor, rode 'pip install skl2onnx onnxruntime' e tente novamente.")
    exit()

# --- CONFIGURAÇÃO ---
# Lista dos modelos que você salvou e que são compatíveis com skl2onnx
# (Modelos do Scikit-learn)
modelos_para_converter = [
    "xgb"
]

# Pasta onde os modelos .pkl estão e onde os .onnx serão salvos
pasta_modelos = "models"

# Número de features (colunas) que seu modelo espera. Pelo nosso histórico, são 16.
numero_de_features = 16

# --- SCRIPT DE CONVERSÃO ---
if ONNX_AVAILABLE:
    print(f"\nProcurando modelos na pasta '{pasta_modelos}/'...")

    # Define o formato de entrada que o ONNX espera
    # [None, numero_de_features] significa que o modelo aceita qualquer número de amostras (None),
    # cada uma com 'numero_de_features' colunas.
    initial_type = [('float_input', FloatTensorType([None, numero_de_features]))]

    for nome_modelo in modelos_para_converter:
        caminho_pkl = f'{pasta_modelos}/{nome_modelo}.pkl'
        caminho_onnx = f'{pasta_modelos}/{nome_modelo}.onnx'

        # Verifica se o arquivo .pkl original existe
        if not os.path.exists(caminho_pkl):
            print(f"AVISO: Arquivo '{caminho_pkl}' não encontrado. Pulando este modelo.")
            continue

        print(f"Convertendo '{nome_modelo}'...")
        try:
            # Carrega o modelo a partir do arquivo .pkl
            modelo_carregado = joblib.load(caminho_pkl)

            # Converte o modelo para o formato ONNX
            modelo_onnx = convert_sklearn(modelo_carregado, initial_types=initial_type)

            # Salva o novo modelo no formato .onnx
            with open(caminho_onnx, "wb") as f:
                f.write(modelo_onnx.SerializeToString())
            print(f"-> SUCESSO! Modelo salvo em '{caminho_onnx}'.")

        except Exception as e:
            print(f"  ERRO ao converter o modelo '{nome_modelo}': {e}")

    print("\n--- Processo de Conversão Finalizado ---")
    print("\nAVISO: Lembre-se que modelos como XGBoost, LightGBM e Comitês (VotingClassifier)")
    print("não são convertidos por este script e requerem métodos especiais ou devem ser")
    print("implementados diretamente na aplicação.")