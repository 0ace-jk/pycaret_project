# Imports
import pandas            as pd
import numpy             as np   
import streamlit         as st
import matplotlib.pyplot as plt
import seaborn           as sns
import plotly.express    as px
import os
import xlsxwriter

from io                     import BytesIO
from pycaret.classification import load_model, predict_model, plot_model
from sklearn.metrics        import confusion_matrix, ConfusionMatrixDisplay

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    # O uso do engine='openpyxl' é obrigatório aqui
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'PyCaret', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write("""## Escorando o modelo gerado no pycaret """)
    st.markdown("""
                Carregue o arquivo no formato .ftr ou .csv contendo os dados para a predição. <br>
                Dados **DEMO** podem ser encontrados no repositório do projeto. <br>
                Repositório:
                [credito-demo](https://github.com)
                """,
                unsafe_allow_html=True
                )
    st.markdown("---")
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type = ['csv','ftr'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)
        df_credit['log_renda'] = np.log(df_credit['renda']+1)
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'modelo_final_lightgbm')
            model_saved = load_model(model_path)
            predict = predict_model(model_saved, data=df_credit)

            fig, ax = plt.subplots(figsize=(10, 6))
            cm = confusion_matrix(predict['mau'], predict['prediction_label'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True'])
            disp.plot(ax=ax, cmap='Greens', values_format='d')
            ax.set_title('LGBMClassifier Confusion Matrix')
            ax.grid(False)
            
            a_1, a_2, a_3 = st.columns([1, 3, 1])
            b_1, b_2, b_3 = st.columns([1, 3, 1])
            b_2.pyplot(fig)

        except Exception as e:
            st.write(f"Error loading model: {e}")
            return
        df_xlsx = convert_df_to_excel(predict)
        a_2.download_button(label='📥 Download',
                            data=df_xlsx ,
                            file_name= 'predict.xlsx')

if __name__ == '__main__':
	main()
    