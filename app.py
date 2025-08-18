import gradio as gr
import pandas as pd
import os
import json

# CARREGAMENTO DE DADOS 

models_info = {}
try:
    with open('models/models_info.json', 'r', encoding='utf-8') as f:
        models_info = json.load(f)
    print("Arquivo de informações dos modelos (models/models_info.json) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'models/models_info.json' não foi encontrado.")
except json.JSONDecodeError:
    print("ERRO: O arquivo 'models/models_info.json' contém um erro de formatação (JSON inválido).")

try:
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    print("Conjuntos de teste (X_test, y_test) carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: 'X_test.csv' ou 'y_test.csv' não encontrados na pasta 'data'.")
    X_test, y_test = pd.DataFrame(), pd.DataFrame()

try:
    predictions_df = pd.read_csv('data/predictions.csv')
    print("Arquivo de previsões (predictions.csv) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'predictions.csv' não encontrado na pasta 'data'.")
    predictions_df = pd.DataFrame()

# CARREGAR IMAGENS DOS GRÁFICOS 
images_dir = 'images'
image_paths = []
if os.path.exists(images_dir):
    # Lista apenas arquivos com extensões de imagem comuns
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png'))]
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    print(f"{len(image_paths)} imagens carregadas da pasta '{images_dir}'.")
else:
    print(f"AVISO: A pasta '{images_dir}' não foi encontrada.")


# FUNÇÃO PARA A INTERFACE 

def show_predictions_from_csv(sample_index):
    sample_index = int(sample_index)
    selected_sample_df = X_test.iloc[[sample_index]].T.reset_index()
    selected_sample_df.columns = ["Feature", "Valor"]
    true_label = int(y_test.iloc[sample_index].iloc[0])
    results = predictions_df.iloc[[sample_index]].T.reset_index()
    results.columns = ["Modelo", "Previsão"]
    return selected_sample_df, results, true_label


# INTERFACE 

with gr.Blocks(theme=gr.themes.Soft(), title="Painel de Análise de Modelos") as demo:
    gr.Markdown("# Painel de Análise de Modelos de Classificação")

    # Início da Definição das Abas 
    with gr.Tab("Comparativo de Previsões"):
        gr.Markdown("Use o slider na parte inferior para escolher uma amostra do conjunto de teste e comparar os resultados.")
        gr.Markdown("---")
        gr.Markdown("### Amostra Selecionada (Dados de Entrada)")
        selected_example_output = gr.DataFrame(
            headers=["Feature", "Valor"], label="Features", interactive=False
        )
        gr.Markdown("---")
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                gr.Markdown("### Previsão dos Modelos")
                results_output = gr.DataFrame(
                    headers=["Modelo", "Previsão"], label="Previsão de Cada Modelo", interactive=False
                )
            with gr.Column(scale=1):
                gr.Markdown("### Valor Real")
                true_label_output = gr.Label(label="Classe Verdadeira")
        with gr.Row():
            example_slider = gr.Slider(
                minimum=0, maximum=len(X_test) - 1 if not X_test.empty else 0,
                step=1, label="Escolha o Índice da Amostra de Teste", value=25
            )

    with gr.Tab("Gráficos de Métricas"):
        # ABA DE GRÁFICOS 
        gr.Markdown("## Visualização das Métricas de Performance")
        gr.Markdown("Abaixo estão os gráficos gerados a partir da avaliação dos modelos no conjunto de teste.")
        
        gr.Gallery(
            value=image_paths, 
            label="Gráficos de Métricas", 
            show_label=False, 
            columns=[2], 
            object_fit="contain"
        )

    with gr.Tab("Metodologia"):
        # ABA DE METODOLOGIA 
        gr.Markdown("## Metodologia Aplicada")
        gr.Markdown("### Fonte dos Dados")
        gr.Markdown("#### House_16H v2")
        gr.Markdown("""
            Conjunto de dados obtido de OpenML (https://www.openml.org/search?type=data&status=active&id=821).
            Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 22784 cases, 17 continuous attributes.
   
            Trata-se de uma versão binarizada do conjunto de dados House_16H, que contém informações da composição demográfica e estado do mercado imobiliário de regiões, com o objetivo de prever o preço mediano das casas nessas regiões.  
            A versão original do conjunto de dados continha 17 atributos contínuos e 22784 amostras. Nesta versão binarizada, o atributo alvo foi transformado em uma variável categórica com duas classes: 'N' (preço mediano acima do limiar) e 'P' (preço mediano abaixo do limiar). O limiar escolhido foi 50000.
            O conjunto foi construído a partir de dados do Censo dos EUA de 1990, que são em maioria contagens acumuladas a partir de diferentes níveis de pesquisas. Todos os estados foram incluídos, e a maioria das contagens foi convertida em proporções apropriadas.  
            A versão denominada '16H' indica que o conjunto original continha 16 atributos preditores, e a letra 'H' sugere uma dificuldade alta na tarefa de modelagem devido à maior variância ou menor correlação dos inputs com o alvo.
                    """
        )
        
        gr.Markdown("### Preparação dos Dados")
        gr.Markdown("""
            Poucas etapas foram necessárias para preparar os dados, não havia valores ausentes ou inconsistências significativas. 
            A principal mudança foi o mapeamento da classe alvo que possuía 2 classes ('N' e 'P') para valores numéricos (0 e 1) para facilitar o treinamento dos modelos de classificação.
            A outra alteração foi a normalização dos dados, a qual foi utilizada apenas para modelos que requerem normalização, como KNN, SVC, MLP.
            """
        )
        gr.Markdown("### Divisão dos Dados")
        gr.Markdown("""
            O conjunto de dados foi dividido em conjuntos de treino e teste, utilizando 64% dos dados para treino, 16% para validação e 20% para teste, estratificada de acordo com a variável alvo.
            Os hiperparâmetros dos modelos foram escolhidos utilizando o conjunto de validação, e a avaliação final foi realizada no conjunto de teste.
            Os modelos finais (os quais vemos as previsões na aba 'Comparativo de Previsões') foram treinados com o conjunto de treino + validação (80%), utilizando os parâmetros obtidos e validados na etapa anterior.
        """)

        gr.Markdown("### Busca de Hiperparâmetros")
        gr.Markdown("""
            Para cada modelo, foi definido um espaço de busca de hiperparâmetros, levando em consideração as limitações computacionais e o tempo disponível.
            A busca foi realizada utilizando validação cruzada com 5 folds, e a métrica de avaliação utilizada foi a f1, o algoritmo de busca foi o random search (RandomizedSearchCV).
            A escolha do random search se deu pela sua eficiência em encontrar boas combinações de hiperparâmetros em menos tempo, especialmente em espaços de busca grandes.
""")

        # PAREI AQUI, SEÇÃO DE MODELOS
        gr.Markdown("### Modelos Utilizados")
        gr.Markdown("""
            Foram treinados e avaliados diversos modelos de classificação.                
            Cada modelo foi ajustado utilizando validação cruzada e otimização de hiperparâmetros para maximizar a acurácia no conjunto de validação.
        """)

        for model_name, info in models_info.items():
            with gr.Accordion(model_name, open=False): # 'open=False' faz com que comecem fechados
                # Mostra a descrição textual
                gr.Markdown(info['description'])
                
                # Adiciona um subtítulo para os hiperparâmetros
                gr.Markdown("#### Hiperparâmetros Utilizados")
                
                # Mostra os hiperparâmetros utilizados
                gr.JSON(value=info['search_space'], label="Espaço de busca")

                # Mostra o dicionário de hiperparâmetros de forma visual
                gr.JSON(value=info['params'], label="Estrutura do Modelo")

        # Você pode adicionar mais texto geral sobre a metodologia aqui, se quiser
        gr.Markdown("---")

    # Eventos
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