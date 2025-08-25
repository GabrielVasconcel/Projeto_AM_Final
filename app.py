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
        with gr.Row():
            example_slider = gr.Slider(
                minimum=0, maximum=len(X_test) - 1 if not X_test.empty else 0,
                step=1, label="Escolha o Índice da Amostra de Teste", value=25
            )
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                gr.Markdown("### Previsão dos Modelos")
                results_output = gr.DataFrame(
                    headers=["Modelo", "Previsão"], label="Previsão de Cada Modelo", interactive=False
                )
            with gr.Column(scale=1):
                gr.Markdown("### Valor Real")
                true_label_output = gr.Label(label="Classe Verdadeira")


    with gr.Tab("Metodologia"):
        # ABA DE METODOLOGIA 
        gr.Markdown("# CRISP-DM: Metodologia de Desenvolvimento")
        gr.Markdown("## Entendimento do Negócio")
        gr.Markdown("### Contextualização")
        gr.Markdown("""O mercado imobiliário é altamente competitivo e vem crescendo cada vez mais nos últimos 
                    anos, trazendo com isso a necessidade de se refinarem as estratégias para se manter nele. A 
                    atribuição de preço a casas e apartamentos é uma prática que apesar de se debruçar sobre uma 
                    base objetiva dos dados referentes àquele imóvel, ainda se utilizade uma subjetividade 
                    humana, além da sazonalidade e oscilações do contexto do momento. Desde área, quantidade 
                    de cômodos, até a localização, vizinhança, ou a conjuntura econômica são atributos que 
                    influenciam no seu valor, e alguns desses atributos podem variar de acordo com a 
                    subjetividade de quem compra. Dessa forma, identificando a natureza multifatorial deste 
                    problema podemos inferir que trata-se de uma situação que pode se beneficiar de modelos de 
                    aprendizagem de máquina para tratar essa grande quantidade de atributos diferentes que o 
                    compõem e assim se tornar uma ferramenta efetiva para o problema.
                    Este projeto visa desenvolver um modelo de classificação para prever o preço mediano de casas em regiões dos EUA, utilizando dados demográficos e do mercado imobiliário. 
                    O objetivo é auxiliar na tomada de decisões relacionadas a investimentos imobiliários e políticas habitacionais, fornecendo previsões precisas sobre o valor das propriedades.
                    """)
        gr.Markdown("### Objetivos do Negócio")
        gr.Markdown("""Para entender as metas estabelecidas é preciso saber que o dataset utilizado tem um target 
                    binário, ou seja, trata-se de um problema de classificação. Então, o objetivo do negócio é 
                    criar modelos de aprendizagem capazes de forma satisfatória, afim de demonstrar que as 
                    ferramentas de inteligência artificial podem fazer parte do ecossistema imobiliário. 
                    """)
        gr.Markdown("### Desafios e Limitações")
        gr.Markdown("""Por se tratar de um problema multifatorial e discretizado a partir de uma base de dados, não 
                    temos as descrições de cada atributo. Portanto, o entendimento dos resultados surgirá a partir 
                    da avaliação do desempenho dos modelos, e não de análises dos dados. Dado que a base de 
                    dados é grande e com muitos atributos, e também, como descrita na documentação, 
                    possivelmente com alta variância e baixa correlação entre atributos, é possível que isso exija 
                    uma complexidade maior do modelo aplicado, o que pode resultar em uma baixa performance 
                    caso essa complexidade não seja atingida. 
                    """)
        gr.Markdown("## Critérios de Sucesso do Negócio")
        gr.Markdown("""Atingir uma acurácia de 90% na classificação dos imóveis. Esse valor demonstra uma 
                    performance suficientemente boa para auxiliar usuários na identificação de imóveis caros, o 
                    que pode trazer maior segurança nos negócios.
                    """)
        gr.Markdown("### Inventário de Recursos")
        gr.Markdown("""O projeto contará com uma base de dados imobiliária do census governamental dos Estados 
                    Unidos, a qual possui 16 atributos e um rótulo binário criado a partir do preço. No quesito 
                    computacional, o desenvolvimento se beneficiará do uso da linguagem de programação 
                    Python e suas bibliotecas para manipulação de dados e criação de modelos de inteligência 
                    artificial, como pandas, Scikitearn e PyTorch. Além disso, a utilização do ambiente de 
                    desenvolvimento colaborativo Google Colab para trabalho em equipe, implementação dos 
                    modelos e treinamento utilizando gpu. 
                    """)
        gr.Markdown("### Requisitos")
        gr.Markdown("""Implementação de um modelo de classificação binária para precificação de imóveis e sua 
                    avaliação por métricas como acurácia, f1-score, precision, recall. 
                    Pressuposições.  O problema é resolvível por modelos de aprendizado de máquina ou não 
                    excede a complexidade que o poder computacional disponível nos permite utilizar. Os dados 
                    são representativos da realidade.
                    """)
        gr.Markdown("### Restrições")
        gr.Markdown("""A capacidade computacional disponível é limitada, o que pode restringir o uso de hiperparâmetros mais robustos.""")
        gr.Markdown("### Riscos")
        gr.Markdown("""Baixa performance do modelo como reflexo de uma complexidade alta no problema. Baixa 
                    capacidade de generalização por conta do dataset não ser representativo da realidade.
                    """)
        gr.Markdown("### Objetivos da Ciência de Dados")
        gr.Markdown("""O objetivo da ciência de dados neste projeto é ser capaz de identificar padrões nos dados que 
                    nos permitam classificar entradas em uma das duas classes do problema, através da criação, 
                    treinamento e testes de modelos de aprendizagem de máquina. Dessa forma, proporcionando 
                    uma ferramenta computacional capaz de diminuir a subjetividade na precificação no mercado 
                    imobiliário.
                    """)
        gr.Markdown("### Critérios de Sucesso da Ciência de Dados")
        gr.Markdown("#### *Acurácia do Modelo*")
        gr.Markdown(""" O modelo deve atingir pelo menos 90% de acurácia na 
                    classificação dos imóveis, conforme definido nos critérios de sucesso do negócio.""") 
        gr.Markdown("#### *Equilíbrio entre Precisão e Recall*")
        gr.Markdown(""" Métricas como F1-score devem ser avaliadas 
                    para garantir que a classificação não esteja enviesada para uma das classes.""")
        gr.Markdown("#### *Generalização*")
        gr.Markdown(""" O modelo deve ser capaz de fazer previsões consistentes em 
                    diferentes subconjuntos dos dados, evitando overfitting.
                    """)
        gr.Markdown("#### *Eficiência Computacional*")
        gr.Markdown(""" O tempo de treinamento e inferência do modelo deve ser 
                    viável para implementação prática.
                    """)
        gr.Markdown("### Plano de Projeto")
        gr.Markdown("O projeto será desenvolvido seguindo as seguintes etapas:")
        gr.Markdown("#### *Planejamento e análise do problema*")
        gr.Markdown(""" Fase inicial do projeto que visa entender o 
                    contexto, listar as restrições e possibilidades, conhecer as ferramentas e técnicas disponíveis e 
                    estabelecer metas. 
                    """)
        gr.Markdown("#### *Análise dos dados*")
        gr.Markdown(""" Fase na qual se realiza um mergulho no dataset do problema 
                    visando entender melhor a realidade do que será trabalhado. Nesta fase já iremos realizar as 
                    manipulações necessárias para que os dados estejam no formato ideal para o treinamento dos 
                    modelos. 
                    """)
        gr.Markdown("#### *Modelagem e treinamento*")
        gr.Markdown("""  Fase onde escolhemos, modelamos e treinamos o modelo. 
                    Métodos de escolha de hiperparâmetros e o uso de métricas para acompanhamento do 
                    desenvolvimento dos modelos serão utilizados para acompanhamento.
                    """)
        gr.Markdown("#### *Avaliação e documentação*")
        gr.Markdown(""" Fase onde analisamos o desempenho final obtido no 
                    conjunto de teste. Além disso será realizada a documentação da fase de desenvolvimento 
                    afim de entender os erros e acertos no processo.
                    """)
        gr.Markdown("### Ferramentas e Técnicas previstas")
        gr.Markdown("""Python (linguagem principal), Pandas, NumPy, Scikit-learn, PyTorch, 
                    Matplotlib e Seaborn. Técnicas: Regularização, cross-validation, gridsearch.
                    """)

    with gr.Tab("Dados"):
        gr.Markdown("## Fonte dos Dados")
        gr.Markdown("### House_16H v2")
        gr.Markdown("""
            Conjunto de dados obtido de OpenML (https://www.openml.org/search?type=data&status=active&id=821).
            Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 22784 cases, 17 continuous attributes.
   
            Trata-se de uma versão binarizada do conjunto de dados House_16H, que contém informações da composição demográfica e estado do mercado imobiliário de regiões, com o objetivo de prever o preço mediano das casas nessas regiões.  
            A versão original do conjunto de dados continha 17 atributos contínuos e 22784 amostras. Nesta versão binarizada, o atributo alvo foi transformado em uma variável categórica com duas classes: 'N' (preço mediano acima do limiar) e 'P' (preço mediano abaixo do limiar). O limiar escolhido foi 50000.
            O conjunto foi construído a partir de dados do Censo dos EUA de 1990, que são em maioria contagens acumuladas a partir de diferentes níveis de pesquisas. Todos os estados foram incluídos, e a maioria das contagens foi convertida em proporções apropriadas.  
            A versão denominada '16H' indica que o conjunto original continha 16 atributos preditores, e a letra 'H' sugere uma dificuldade alta na tarefa de modelagem devido à maior variância ou menor correlação dos inputs com o alvo.
                    """
        )
        
        gr.Markdown("## Preparação dos Dados")
        gr.Markdown("""
            Poucas etapas foram necessárias para preparar os dados, não havia valores ausentes ou inconsistências significativas. 
            A principal mudança foi o mapeamento da classe alvo que possuía 2 classes ('N' e 'P') para valores numéricos (0 e 1) para facilitar o treinamento dos modelos de classificação.
            A outra alteração foi a normalização dos dados, a qual foi utilizada apenas para modelos que requerem normalização, como KNN, SVC, MLP.
            """
        )
        gr.Markdown("## Divisão dos Dados")
        gr.Markdown("""
            O conjunto de dados foi dividido em conjuntos de treino e teste, utilizando 64% dos dados para treino, 16% para validação e 20% para teste, estratificada de acordo com a variável alvo.
            Os hiperparâmetros dos modelos foram escolhidos utilizando o conjunto de validação, e a avaliação final foi realizada no conjunto de teste.
            Os modelos finais (os quais vemos as previsões na aba 'Comparativo de Previsões') foram treinados com o conjunto de treino + validação (80%), utilizando os parâmetros obtidos e validados na etapa anterior.
        """)
        gr.Markdown("# Análise Exploratória dos Dados")
        gr.Markdown("## Relatório inicial da coleta de dados")
        gr.Markdown("""Os dados foram obtidos a partir do site OpenML disponibilizados por Joaquin Vanschoren 
                    sendo a segunda versão de outro dataset. Os dados originais foram retirados do US Census 
                    Bureau de uma coleta que fez parte do censo de 1990 dos Estados Unidos. A versão 2 deste 
                    dataset, difere da versão anterior apenas em relação ao atributo “Price”, preço, o qual foi 
                    transformado em uma classe que divide em positivo e negativo a partir de um valor arbitrário 
                    escolhido pelo desenvolvedor dos dados. Todas os atributos estão discretizados e não foram 
                    fornecidas descrições sobre eles na documentação.
                    """)
        gr.Markdown("## Descrição dos Dados")
        gr.Markdown("""O conjunto conta com 17 colunas, sendo 16 features e 1 target baseado em preço, e possui 
                    22784 exemplos. Não há valores ausentes na tabela. Todas as features são valores de ponto 
                    flutuante e o target é binário. As informações originais foram transformadas em proporções 
                    adequadas para este dataset, portanto encontram-se num alcance geral de 0 a 1 
                    aproximadamente, com exceção da coluna “P1” que possui valores entre 2 e 7 milhões.
                    """)
        gr.Markdown("## Exploração dos Dados")
        gr.Markdown("""Utilizamos a plotagem em boxplot para visualizar a distribuição de cada atributo e 
                    identificamos que a maior parte das colunas possuía uma grande quantidade de outliers.
                    """)
        gr.Image(label="Boxplots dos Atributos", value="images/boxplot.png", show_label=False, width=800)  
        gr.Markdown("""Isso evidencia que as colunas têm uma alta densidade próximo a determinado valor porém 
                    ainda existem valores que divergem dessa parte mais densa dos dados, ou seja a calda da 
                    distribuição é longa e o pico é alto. A alta presença de outliers, aliada ao fato de que sabemos 
                    que os dados originais eram contagens e foram transformados em proporções nos indica que a 
                    remoção desses valores não é a abordagem ideal.
                    """)
        gr.Image(label="Histograma H2p2", value="images/H2p2_hist.png", show_label=False, width=800)
        gr.Markdown("""Pegando a coluna “H2p2” como exemplo, um histograma ilustra de forma clara como há a 
                    concentração da maior parte dos valores em um alcance pequeno de 0 a 0.2, e ainda 
                    mantém-se tendo valores até 1. Essa mesma dinâmica foi observada na grande maioria das 
                    colunas do dataset. Ao realizar uma Análise de Componentes Principais (PCA), observamos 
                    que entre 8 e 10 componentes explicam aproximadamente 100% da variância dos dados. Isso 
                    indica que é possível reduzir a dimensionalidade do problema para um conjunto de 8 a 10 
                    variáveis sem perda significativa de informação. Essa redução pode beneficiar a fase de 
                    criação e teste de modelos de classificação, pois diminui a complexidade dos dados, 
                    reduzindo o risco de sobreajuste (overfitting) e melhorando a eficiência computacional, sem 
                    comprometer a performance preditiva.    
                    """)
        gr.Image(label="PCA - Explained Variance", value="images/explained_variance_pca.png", show_label=False, width=800)
        gr.Markdown("""A partir de uma análise da versão dos dados utilizadas no projeto v2, e a versão anterior, 
                    conseguimos identificar o limiar utilizado para separar as classes no problema, o que não foi 
                    explicado na documentação do conjunto. A classe target foi dividida da seguinte forma, 
                    valores menores ou iguais a 50000 foram mapeados para a classe positiva “P”, enquanto que 
                    valores maiores se tornaram a classe negativa “N”. Esta é a conclusão mais provável, 
                    entretanto não podemos afirmar com certeza, dado que ao realizar a análise encontramos que 
                    uma das classe tinha o valor máximo 50000 e a outra o valor mínimo de 50100, estando o 
                    limiar dentro desse alcance supomos 50000 por conveniência. 
                    """)
        gr.Markdown("## Qualidade dos Dados")
        gr.Markdown("""A ausência de dados faltantes é um fator positivo observado no dataset, uma vez que a etapa 
                    de tratamento desse tipo de problema não precisou ser implementada. A documentação 
                    deixou claro que foram realizadas alterações nos dados, entretanto não informou exatamente 
                    que tipo foi executado, apenas alegou ter transformado contagens em proporções adequadas. 
                    A falta de descrições na documentação sobre o significado de cada atributo é um fator que 
                    pode atrapalhar o processo de treinamento e aprendizagem do modelo, visto que ter 
                    informações sobre as colunas ajudaria em ter uma ideia do que influencia o desempenho do 
                    modelo e assim ter o entendimento e capacidade de explicação do problema na realidade. 
                    Apesar de afirmar ter realizado alterações, acreditamos que uma das colunas, a “P1”, não foi 
                    alvo delas, mas a conclusão dessa suposição não pode ser feita dado que não sabemos o que a 
                    coluna representa. A grande presença de outliers não é um problema, e a partir da análise que 
                    fizemos, acreditamos que o sucesso do projeto depende em grande parte da forma como 
                    iremos lidar com eles. O dataset carece de uma descrição detalhada e diverge um pouco das 
                    afirmações feitas sobre ele na documentação. Apesar disso, acreditamos que ele tem 
                    qualidade o suficiente para que o projeto seja realizado e que as metas definidas sejam 
                    atingidas de forma satisfatória.
                    """)



    with gr.Tab("Modelos e Hiperparâmetros"):
        gr.Markdown("### Busca de Hiperparâmetros")
        gr.Markdown("""
            Para cada modelo, foi definido um espaço de busca de hiperparâmetros, levando em consideração as limitações computacionais e o tempo disponível.
            A busca foi realizada utilizando validação cruzada com 5 folds, e a métrica de avaliação utilizada foi a f1, o algoritmo de busca foi o random search (RandomizedSearchCV).
            A escolha do random search se deu pela sua eficiência em encontrar boas combinações de hiperparâmetros em menos tempo, especialmente em espaços de busca grandes.
            """)

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

    with gr.Tab("Avaliação Comparativa"):
        # ABA DE GRÁFICOS 
        gr.Markdown("""Para compreender melhor o desempenho dos modelos selecionados, foram gerados gráficos 
                    comparativos exibindo os valores das principais métricas no conjunto de teste. Esses gráficos 
                    de barras permitem uma visualização clara das diferenças entre os modelos finais, facilitando 
                    a identificação de pontos fortes e fracos de cada um. Através dessa análise visual, podemos 
                    avaliar como cada modelo equilibra acurácia, f1-score, recall, precisão e AUC, além de 
                    entender quais técnicas se mostraram mais eficazes na tarefa proposta. 
                    """)
        gr.Image(label="Comparativo de Acurácia", value="images/acuracia.png", show_label=False, width=800)        
        gr.Markdown("""O gráfico acima mostra a acurácia de cada modelo no conjunto de teste. 
                    Os modelos MLP, Comitê de MLPs, Xgboost e Light GBM se destacaram com as maiores acurácias.
                    """)
        gr.Image(label="Comparativo de f1-score", value="images/f1.png", show_label=False, width=800)        
        gr.Markdown("""O gráfico acima mostra o f1 score de cada modelo no conjunto de teste. 
                    Os modelos MLP, Comitê de MLPs, Xgboost e Light GBM se destacaram com os maiores valores.
                    """)
        gr.Image(label="Comparativo de Precision", value="images/precisao.png", show_label=False, width=800)        
        gr.Markdown("""O gráfico acima mostra a precisão de cada modelo no conjunto de teste. 
                    Os modelos MLP, Comitê de MLPs, Xgboost e Light GBM se destacaram com os maiores valores.
                    """)
        gr.Image(label="Comparativo de AUC", value="images/auc.png", show_label=False, width=800)        
        gr.Markdown("""O gráfico acima mostra a AUC de cada modelo no conjunto de teste. 
                    Os modelos MLP, Comitê de MLPs, Comitê heterogêneno e Light GBM se destacaram com os maiores valores.
                    """)
        gr.Image(label="Comparativo de recall", value="images/recall.png", show_label=False, width=800)        
        gr.Markdown("""O gráfico acima mostra a recall de cada modelo no conjunto de teste. 
                    Os modelos MLP, Comitê de MLPs, Xgboost e KNN se destacaram com os maiores valores.
                    """)
        gr.Image(label="Comparativo Geral", value="images/metricas_geral.png", show_label=False, width=800)        
        gr.Markdown("""Os modelos KNN, Árvore de Decisão, Random Forest e SVC apresentaram desempenho 
                    inferior em comparação aos outros devido a diferentes limitações inerentes às suas 
                    abordagens:

                    ### KNN
                    
                    Apesar de sua simplicidade e eficácia em certos cenários, o KNN tende a sofrer 
                    com alta variância e sensibilidade a ruído, especialmente em conjuntos de dados 
                    maiores e mais complexos. Além disso, seu desempenho pode ser prejudicado se os 
                    dados não possuírem uma separação bem definida no espaço vetorial. 
                    ### Árvore de Decisão
                    
                     Embora interpretável, esse modelo frequentemente sofre de 
                    overfitting, aprendendo demasiadamente os padrões do conjunto de treino e falhando 
                    em generalizar bem para novos dados. 
                    ### Random Forest
                     
                    Como um conjunto de múltiplas árvores de decisão, esse modelo é 
                    mais robusto que uma única árvore, mas ainda assim pode ser limitado em cenários 
                    onde a complexidade dos padrões exige modelos mais sofisticados. Além disso, pode 
                    apresentar menor capacidade de captura de relações complexas nos dados em 
                    comparação com modelos mais avançados, como redes neurais e boosting. 
                    ### SVC (Support Vector Classifier)
                     
                    O SVC pode ser muito eficaz em espaços de baixa 
                    dimensionalidade, mas pode ter dificuldades em conjuntos de dados grandes, 
                    especialmente quando os dados não são perfeitamente separáveis. Seu desempenho 
                    também depende fortemente da escolha do kernel e da escala dos dados, o que pode 
                    torná-lo menos eficiente em comparação a modelos como XGBoost e LightGBM, que 
                    conseguem capturar padrões complexos com maior flexibilidade. 
                    
                    No geral, os modelos baseados em boosting e comitês apresentaram um melhor equilíbrio 
                    entre generalização e robustez, o que explica sua superioridade nos resultados finais.
                    """)
        gr.Markdown("## Seleção dos 4 melhores modelos")
        gr.Markdown("""Para selecionar os quatro melhores modelos, foi aplicada uma média ponderada considerando 
                    diferentes pesos para as métricas de avaliação: acurácia (1), f1-score (2.5), recall (2), precisão 
                    (1.5) e AUC (2). Essa abordagem valoriza mais o f1-score, seguido do recall e AUC, 
                    refletindo a importância de modelos que não apenas acertam a classificação, mas também 
                    garantem um bom equilíbrio entre precisão e recall, além de entender a natureza dos dados e 
                    seu desbalanceamento entre classes. 
                    Os quatro melhores modelos selecionados com essa estratégia foram: 
                    """)
        gr.Markdown(""" 
                    ### LGBM 
                    
                    Média ponderada mais alta, com destaque para recall (0.9261), f1-score 
                    (0.9227) e AUC (0.9768), indicando alta capacidade preditiva e boa separação entre 
                    classes. 
                    """)
        gr.Image(label="Matriz de Confusão LGBM", value="images/cm_lgbm.png", show_label=False, width=600)
        gr.Markdown("""
                    ### Comitê de MLPs 
                    
                    Desempenho muito próximo ao LGBM, com f1-score (0.9235) e 
                    recall (0.9292) elevados, além de um AUC de 0.9762.
                    """)
        gr.Image(label="Matriz de Confusão Comitê de MLPs", value="images/cm_comiteMLP.png", show_label=False, width=600) 
        gr.Markdown(""" 
                    ### MLP 
                    
                    Fica ligeiramente abaixo do Comitê de MLPs, mantendo um equilíbrio entre 
                    todas as métricas, especialmente recall (0.9218) e AUC (0.9744). 
                    """)
        gr.Image(label="Matriz de Confusão MLP", value="images/cm_mlp.png", show_label=False, width=600)
        gr.Markdown("""
                    ### Comitê Heterogêneo 
                    
                    Apesar de ter uma acurácia um pouco menor (0.8813), o 
                    modelo se destacou pelo recall (0.9349) e um AUC competitivo (0.9659), garantindo 
                    boa separação de classes. 
                    """)
        gr.Image(label="Matriz de Confusão Comitê Heterogêneo", value="images/cm_comiteHET.png", show_label=False, width=600)
        gr.Markdown("""     
                    Comparando esses resultados, os modelos LGBM e Comitê de MLPs são os mais fortes, 
                    possivelmente devido à combinação de técnicas de aprendizado mais sofisticadas. O MLP 
                    isolado também apresentou um desempenho robusto, o que indica que modelos de redes 
                    neurais foram bem ajustados. Já o Comitê Heterogêneo se saiu bem em recall, garantindo boa 
                    recuperação de instâncias positivas. 
                    A análise gráfica desses modelos permite uma comparação visual dos desempenhos 
                    ponderados, destacando suas diferenças em cada métrica e confirmando quais se saíram 
                    melhor em aspectos específicos. 
                    """)
        gr.Image(label="Comparativo dos 4 Melhores Modelos", value="images/top4.png", show_label=False, width=800)
        
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