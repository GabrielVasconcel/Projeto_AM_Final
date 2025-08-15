import gradio as gr
import pandas as pd
import os

# --- 1. CARREGAMENTO DE DADOS ---

# Carregar os dados de teste e o valor real
try:
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    print("Conjuntos de teste (X_test, y_test) carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: 'X_test.csv' ou 'y_test.csv' não encontrados na pasta 'data'.")
    X_test, y_test = pd.DataFrame(), pd.DataFrame()

# Carregar o arquivo com as previsões pré-calculadas
try:
    # Este DataFrame terá colunas com os nomes dos modelos
    predictions_df = pd.read_csv('data/predictions.csv')
    print("Arquivo de previsões (predictions.csv) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'predictions.csv' não encontrado na pasta 'data'.")
    predictions_df = pd.DataFrame()

# --- 2. FUNÇÃO PARA A INTERFACE GRADIO ---

def show_predictions_from_csv(sample_index):
    """
    Busca os dados e as previsões com base no índice e os retorna para a interface.
    """
    sample_index = int(sample_index)

    # 1. Pega a amostra de entrada (features) do X_test
    selected_sample_df = X_test.iloc[[sample_index]].T.reset_index()
    selected_sample_df.columns = ["Feature", "Valor"]

    # 2. Pega o valor real e CONVERTE para um int padrão do Python
    true_label = int(y_test.iloc[sample_index].iloc[0]) 

    # 3. Pega as previsões dos modelos do predictions_df
    results = predictions_df.iloc[[sample_index]].T.reset_index()
    results.columns = ["Modelo", "Previsão"]

    return selected_sample_df, results, true_label


# --- 3. CONSTRUÇÃO DA INTERFACE GRADIO (SLIDER EMBAIXO) ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Painel de Comparação de Modelos de Classificação")
    gr.Markdown("Use o slider na parte inferior para escolher uma amostra do conjunto de teste e comparar os resultados.")

    gr.Markdown("---")
    
    # --- BLOCO SUPERIOR: DADOS DE ENTRADA ---
    gr.Markdown("### Amostra Selecionada (Dados de Entrada)")
    selected_example_output = gr.DataFrame(
        headers=["Feature", "Valor"], 
        label="Features", 
        interactive=False
    )

    gr.Markdown("---")

    # --- BLOCOS INTERMEDIÁRIOS: PREVISÕES E VALOR REAL LADO A LADO ---
    with gr.Row(variant="panel"):
        # Coluna da Esquerda para a tabela de previsões
        with gr.Column(scale=2): 
            gr.Markdown("### Previsão dos Modelos")
            results_output = gr.DataFrame(
                headers=["Modelo", "Previsão"], 
                label="Previsão de Cada Modelo", 
                interactive=False
            )
        
        # Coluna da Direita para o valor real
        with gr.Column(scale=1):
            gr.Markdown("### Valor Real")
            true_label_output = gr.Label(label="Classe Verdadeira")

    # --- CONTROLE PRINCIPAL (MOVIDO PARA O FINAL) ---
    with gr.Row():
        example_slider = gr.Slider(
            minimum=0,
            maximum=len(X_test) - 1 if not X_test.empty else 0,
            step=1,
            label="Escolha o Índice da Amostra de Teste",
            value=25
        )

    # Os eventos não mudam de lugar, eles apenas referenciam os componentes
    example_slider.change(
        fn=show_predictions_from_csv,
        inputs=example_slider,
        outputs=[selected_example_output, results_output, true_label_output]
    )
    
    demo.load(
        fn=show_predictions_from_csv,
        inputs=example_slider,
        outputs=[selected_example_output, results_output, true_label_output]
    )


demo.launch()