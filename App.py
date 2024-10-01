import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb
import plotly.express as px

st.set_page_config(layout="wide")

# Carregar o modelo treinado

def load_model():
    with open('modelo_xgb_doenca_cardiaca.pkl', 'rb') as file:
        modelo = pickle.load(file)
    return modelo

modelo = load_model()

# Função para mapear entradas de Sim/Não para 1/0
def map_sim_nao(valor):
    return 1 if valor == 'Sim' else 0

# Função para mapear entradas de raça
def map_raca(valor):
    if valor == 'Branco(a)':
        return 'White'
    elif valor == 'Negro(a)':
        return 'Black'
    elif valor == 'Asiático(a)':
        return 'Asian'
    elif valor == 'Outro(a)':
        return 'Other'
    elif valor == 'Latino(a)':
        return 'Hispanic'
    else:
        return None

# Função para mapear entradas de diabetes
def map_diabetes(valor):
    if valor == 'Sim':
        return 'Yes'
    elif valor == 'Não':
        return 'No'
    elif valor == 'Pré-diabetes':
        return 'No, borderline diabetes'
    elif valor == 'Sim, (durante a gravidez)':
        return 'Yes (during pregnancy)'
    else:
        return None

# Função para mapear entradas de saúde geral
def map_saude_geral(valor):
    if valor == 'Muito boa':
        return 'Very good'
    elif valor == 'Normal':
        return 'Fair'
    elif valor == 'Boa':
        return 'Good'
    elif valor == 'Ruim':
        return 'Poor'
    elif valor == 'Excelente':
        return 'Excellent'
    else:
        return None

# Função para injetar CSS personalizado
def add_css_style():
    st.markdown(
        """
        <style>
        h1 {
            font-size: 50px;
        }
        p, ul, li {
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Chama a função para injetar CSS
add_css_style()


# Título da aplicação

st.title('Previsão de Doenças Cardíacas com XGBoost')

st.write('''
Este aplicativo web foi desenvolvido para prever a probabilidade de um paciente desenvolver doenças cardíacas utilizando XGBoost, uma poderosa biblioteca de machine learning. O modelo foi treinado com dados obtidos do Kaggle, garantindo uma base de dados ampla e confiável para análises precisas.
''')

st.subheader('Principais Características:')
st.write('''
- **Modelo Avançado**: Utiliza o algoritmo XGBoost, conhecido por sua eficiência e alto desempenho em tarefas de classificação.
- **Dados Abrangentes**: As informações consideradas incluem Índice de Massa Corporal (IMC), hábitos de fumar, consumo de álcool, histórico de AVC, saúde física e mental, entre outros fatores relevantes.
- **Alta Precisão**: O modelo apresenta um Recall de 80%, proporcionando previsões confiáveis sobre o risco de doenças cardíacas.
''')

st.subheader('Como Funciona:')
st.write('''
- **Inserção de Dados**: O usuário preenche as informações do paciente, selecionando opções como "Sim" ou "Não" para hábitos de saúde, categorias de idade, raça, entre outros.
- **Previsão**: Ao clicar no botão "Prever", o modelo calcula a probabilidade de o paciente apresentar problemas cardíacos.
- **Resultado**: O aplicativo exibe de forma clara se o paciente possui ou não risco de desenvolver doenças cardíacas, acompanhada da probabilidade calculada.
''')

st.subheader('Considerações Importantes:')
st.write('''
- **Ferramenta Auxiliar**: Este aplicativo serve como uma ferramenta de apoio para avaliação de riscos e não substitui uma consulta médica profissional.
- **Privacidade dos Dados**: As informações inseridas no aplicativo são utilizadas exclusivamente para a previsão e não são armazenadas ou compartilhadas.
- **Atualizações Futuras**: Estamos continuamente aprimorando o modelo e a interface do aplicativo para oferecer previsões ainda mais precisas.
''')

st.subheader('Tecnologias Utilizadas:')
st.write('''
- **XGBoost**: Para construção e treinamento do modelo de previsão.
- **Streamlit**: Para desenvolvimento da interface web interativa.
- **Pandas**: Para manipulação e processamento dos dados.
- **Pickle**: Para serialização e carregamento do modelo treinado.
- **Plotly**: Para criação de gráficos
''')

st.title('Insira as informações do paciente')

# Coletando os inputs do usuário
# IMC
imc = st.number_input('Índice de Massa Corporal (IMC)', min_value=10.0, value=10.0, step=0.1)


# Fumante
fumante = st.selectbox('Fumante', ['Não', 'Sim'], index=1 if 1 else 0)
fumante = map_sim_nao(fumante)

# Consumo de álcool
consumo_alcool = st.selectbox('Consumo de Álcool', ['Não', 'Sim'], index=0)
consumo_alcool = map_sim_nao(consumo_alcool)

# Derrame (AVC)
derrame = st.selectbox('Acidente Vascular Cerebral (AVC)', ['Não', 'Sim'], index=0)
derrame = map_sim_nao(derrame)

# Saúde física
saude_fisica = st.number_input('Número de dias em que teve doenças e lesões físicas nos últimos 30 dias',  min_value= 0, value= 0, step= 1)

# Saúde mental
saude_mental = st.number_input('Número de dias em que NÃO teve a saúde mental boa nos últimos 30 dias',  min_value= 0, value= 0, step= 1)

# Dificuldade para Caminhar
dificuldades_caminhar = st.selectbox('Dificuldades para caminhar ou subir escadas', ['Não', 'Sim'], index=0)
dificuldades_caminhar = map_sim_nao(dificuldades_caminhar)

# sexo
sexo = st.selectbox('sexo', ['Feminino', 'Masculino'], index=0)
sexo = 0 if sexo == 'Feminino' else 1

# Idade
idade = st.selectbox(
    'Idade',
    ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', 
     '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29'],
    index=0
)

# Raça
raca = st.selectbox(
    'Raça',
    ['Branco(a)', 'Negro(a)', 'Asiático(a)', 'Outro(a)', 'Latino(a)'],
    index=0)
raca = map_raca(raca)

# Diabetes
diabetes = st.selectbox(
    'Diabetes',
    ['Sim', 'Não', 'Pré-diabetes', 'Sim, (durante a gravidez)'],
    index=1)
diabetes = map_diabetes(diabetes)

# Atividade física
atividade_fisica = st.selectbox('Atividade física ou exercício nos últimos 30 dias', ['Não', 'Sim'], index=0)
atividade_fisica = map_sim_nao(atividade_fisica)

# Saúde geral
saude_geral = st.selectbox(
    'Saúde em Geral',
    ['Muito boa', 'Normal', 'Boa', 'Ruim', 'Excelente'],
    index=2)
saude_geral = map_saude_geral(saude_geral)

# Tempo de sono
tempo_sono = st.number_input('Tempo de Sono (horas)', min_value=0, value=7, step=1)

# Asma
asma = st.selectbox('Asma', ['Não', 'Sim'], index=0)
asma = map_sim_nao(asma)

# Doença renal
doenca_renal = st.selectbox('Doença Renal', ['Não', 'Sim'], index=0)
doenca_renal = map_sim_nao(doenca_renal)

# Câncer de pele
cancer_pele = st.selectbox('Câncer de Pele', ['Não', 'Sim'], index=1)
cancer_pele = map_sim_nao(cancer_pele)

# Botão para previsão
if st.button('Prever'):
    # Criar dicionário com as entradas
    paciente = {
        'BMI': [imc],
        'Smoking': [fumante],
        'AlcoholDrinking': [consumo_alcool],
        'Stroke': [derrame],
        'PhysicalHealth': [atividade_fisica],
        'MentalHealth': [saude_mental],
        'DiffWalking': [dificuldades_caminhar],
        'Sex': [sexo],
        'AgeCategory': [idade],
        'Race': [raca],
        'Diabetic': [diabetes],
        'PhysicalActivity': [atividade_fisica],
        'GenHealth': [saude_geral],
        'SleepTime': [tempo_sono],
        'Asthma': [asma],
        'KidneyDisease': [doenca_renal],
        'SkinCancer': [cancer_pele]
    }

    # Converter para DataFrame
    df = pd.DataFrame(paciente)

    # Converter colunas tipo object para category
    colunas_objeto = df.select_dtypes(include= 'object').columns
    df[colunas_objeto] = df[colunas_objeto].astype('category')


    # Fazer previsão
    previsao = modelo.predict(df)
    probabilidade = modelo.predict_proba(df)[:,1]  # Probabilidade da classe positiva

    st.write(''' Probabilidade de Problema Cardíaco ''')

    # Exibir o resultado
    if previsao[0] == 1:
        st.error(f'O paciente **possui** risco de problemas cardíacos. Probabilidade: {probabilidade[0]*100:.2f}%')
    else:
        st.success(f'O paciente **não possui** risco de problemas cardíacos. Probabilidade: {probabilidade[0]*100:.2f}%')
    st.markdown(
    '<small>**ATENÇÃO!** Este aplicativo serve como uma ferramenta de apoio para avaliação de riscos e não substitui uma consulta médica profissional. Recomenda-se sempre buscar orientação de um especialista para diagnósticos e tratamentos.</small>',
    unsafe_allow_html=True
    )
     # Obter importâncias do modelo
    importancias = modelo.feature_importances_
    colunas = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
        'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
        'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']

    # Criando um DataFrame com importâncias
    df_importancias = pd.DataFrame({
        'Feature': colunas,
        'Importance': importancias
    })

     # Filtrando para as variáveis desejadas
    features_selecionadas = ['DiffWalking', 'GenHealth', 'Stroke', 'Diabetic']
    df_features_selecionadas = df_importancias[df_importancias['Feature'].isin(features_selecionadas)]

    # Traduzindo os nomes das variáveis para português
    df_features_selecionadas = df_features_selecionadas.copy()
    df_features_selecionadas['Feature'] = df_features_selecionadas['Feature'].replace({
        'DiffWalking': 'Dificuldades para Caminhar ou subir escadas',
        'GenHealth': 'Saúde em Geral',
        'Stroke': 'Acidente Vascular Cerebral',
        'Diabetic': 'Diabetes'
        })

    df_features_selecionadas = df_features_selecionadas.sort_values(by='Importance', ascending= True)


     # Criando gráfico com Plotly
    fig = px.bar(
        df_features_selecionadas,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Fatores importantes a se considerar',
        labels={'Importance': 'Grau de intensidade ', 'Feature': 'Fatores'},
        
    )

    if previsao[0] == 1:
        fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',  # Define o fundo da área do gráfico como branco
        'paper_bgcolor': 'rgba(255, 255, 255, 1)', # Define o fundo externo ao gráfico como branco
        'title': {'text': "ATENÇÃO PARA ESSES FATORES!", 'font': {'color': 'black'}},
        'xaxis': {'title': 'Grau de intensidade', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
        'yaxis': {'title': 'Fatores', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    })
        st.plotly_chart(fig)
    else:
        ""

        
