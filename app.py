import gradio as gr
import pandas as pd
import os
import joblib
import numpy as np
import onnxruntime as rt

# --- 1. CARREGAMENTO DE DADOS E MODELOS ---

# Carregar os dados de teste
try:
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    print("Conjuntos de teste (X_test, y_test) carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: 'X_test.csv' ou 'y_test.csv' não encontrados na pasta 'data'.")
    X_test, y_test = pd.DataFrame(), pd.DataFrame()

# Carregar o scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    print("Scaler carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: Arquivo 'models/scaler.pkl' não encontrado.")
    scaler = None

# Carregar os modelos ONNX
model_names = ["knn", "mlp", "svc", "tree", "forest"]
# A lista norm_model_names não é mais necessária, pois todos os modelos usarão dados normalizados.
models = {}
print("\nCarregando modelos ONNX...")
for name in model_names:
    caminho_onnx = f'models/{name}.onnx'
    try:
        models[name] = rt.InferenceSession(caminho_onnx)
        print(f"-> Modelo '{name}.onnx' carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar '{caminho_onnx}': {e}")
print(f"\n{len(models)} modelos ONNX prontos para uso!")


# --- 2. FUNÇÃO PRINCIPAL DE INFERÊNCIA ---

def run_inference_on_sample(example_index):
    """
    Executa todos os modelos para uma única amostra do conjunto de teste.
    """
    if X_test.empty or scaler is None:
        return pd.DataFrame(), pd.DataFrame(), "Erro: Dados ou scaler não carregados."

    example_index = int(example_index)
    selected_example_df = X_test.iloc[[example_index]]
    true_label = y_test.iloc[example_index].iloc[0]

    # --- LÓGICA DE TRANSFORMAÇÃO CORRIGIDA E SIMPLIFICADA ---
    
    # Aplica o scaler no DataFrame completo da amostra.
    # O resultado é um array NumPy que será usado para TODOS os modelos.
    input_data_np = scaler.transform(selected_example_df).astype(np.float32)
    
    # --- FIM DA CORREÇÃO ---

    results = []
    for name, sess in models.items():
        input_name = sess.get_inputs()[0].name
        
        # Todos os modelos agora recebem os mesmos dados normalizados
        prediction_result = sess.run(None, {input_name: input_data_np})
        prediction = prediction_result[0][0]
        
        results.append({"Modelo": name, "Previsão da Classe": prediction})
    
    results_df = pd.DataFrame(results)
    
    return selected_example_df, results_df, str(true_label)


# --- 3. CONSTRUÇÃO DA INTERFACE GRADIO ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Painel de Comparação de Modelos de Classificação")
    gr.Markdown("Use o slider abaixo para escolher uma amostra do conjunto de teste. A aplicação irá executar todos os modelos para essa amostra e mostrar suas previsões, comparando com o valor real.")

    with gr.Row():
        example_slider = gr.Slider(
            minimum=0,
            maximum=len(X_test) - 1 if not X_test.empty else 0,
            step=1,
            label="Escolha o Índice da Amostra de Teste",
            value=25
        )

    gr.Markdown("---")

    with gr.Row(variant="panel"):
        with gr.Column(scale=3):
            gr.Markdown("### Amostra Selecionada (Dados de Entrada)")
            selected_example_output = gr.DataFrame(label="Features")
        
        with gr.Column(scale=1):
            gr.Markdown("### Valor Real")
            true_label_output = gr.Label(label="Classe Verdadeira")

    gr.Markdown("---")
    gr.Markdown("### Resultados da Classificação")
    results_output = gr.DataFrame(label="Previsão de Cada Modelo")

    example_slider.change(
        fn=run_inference_on_sample,
        inputs=example_slider,
        outputs=[selected_example_output, results_output, true_label_output]
    )
    
    demo.load(
        fn=run_inference_on_sample,
        inputs=example_slider,
        outputs=[selected_example_output, results_output, true_label_output]
    )

demo.launch()
