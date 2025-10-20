import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =========================================
# CONFIGURAÇÃO BÁSICA DA PÁGINA
# =========================================
st.set_page_config(
    page_title="Dashboard CCEE ⚡",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard de Energia - CCEE")
st.markdown("Análise exploratória, previsão e insights com base nos dados da CCEE.")

# =========================================
# LEITURA DOS DADOS
# =========================================
@st.cache_data
def carregar_dados():
    df = pd.read_csv("ccee_energia.csv", sep=",")
    
    # Ajuste de colunas (corrige nomes e tipos)
    df.columns = df.columns.str.strip()
    
    # Detecta automaticamente a coluna de data
    for col in df.columns:
        if "data" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df.rename(columns={col: "Data"}, inplace=True)
    
    # Cria colunas auxiliares
    df["Ano"] = df["Data"].dt.year
    df["Mês"] = df["Data"].dt.month
    return df

df = carregar_dados()

# =========================================
# FILTROS
# =========================================
st.sidebar.header("🔍 Filtros")

anos = sorted(df["Ano"].dropna().unique())
ano_sel = st.sidebar.selectbox("Selecione o Ano", anos)

agentes = ["Todos"] + sorted(df["Agente"].dropna().unique()) if "Agente" in df.columns else ["Todos"]
agente_sel = st.sidebar.selectbox("Selecione o Agente", agentes)

submercados = ["Todos"] + sorted(df["Submercado"].dropna().unique()) if "Submercado" in df.columns else ["Todos"]
submercado_sel = st.sidebar.selectbox("Selecione o Submercado", submercados)

# =========================================
# APLICAÇÃO DOS FILTROS
# =========================================
df_filtrado = df[df["Ano"] == ano_sel]

if agente_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Agente"] == agente_sel]

if submercado_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Submercado"] == submercado_sel]

# =========================================
# GRÁFICOS PRINCIPAIS
# =========================================
st.markdown("## 📊 Visão Geral")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Preço Médio de Energia (PLD)")
    if "Preço" in df_filtrado.columns:
        preco_mensal = df_filtrado.groupby("Mês")["Preço"].mean()
        st.line_chart(preco_mensal)
    else:
        st.warning("Coluna 'Preço' não encontrada na base.")

with col2:
    st.subheader("Volume Total Comercializado (MWh)")
    if "Energia_MWh" in df_filtrado.columns:
        energia_mensal = df_filtrado.groupby("Mês")["Energia_MWh"].sum()
        st.bar_chart(energia_mensal)
    else:
        st.warning("Coluna 'Energia_MWh' não encontrada na base.")

# =========================================
# SEÇÃO DE ANÁLISE DETALHADA
# =========================================
st.markdown("---")
st.subheader("📈 Análise Detalhada e Previsão")

if "Preço" in df_filtrado.columns and "Mês" in df_filtrado.columns:
    X = df_filtrado[["Mês"]]
    y = df_filtrado["Preço"]

    modelo = LinearRegression()
    modelo.fit(X, y)
    previsoes = modelo.predict(X)

    mae = mean_absolute_error(y, previsoes)
    rmse = mean_squared_error(y, previsoes, squared=False)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Dados Reais")
    ax.plot(X, previsoes, color="red", label="Previsão Linear")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Preço (R$/MWh)")
    ax.legend()
    st.pyplot(fig)

    st.metric("Erro Médio Absoluto (MAE)", f"{mae:.2f}")
    st.metric("Raiz do Erro Quadrático Médio (RMSE)", f"{rmse:.2f}")
else:
    st.warning("Colunas necessárias para previsão não encontradas na base.")

# =========================================
# TABELA FINAL
# =========================================
st.markdown("---")
st.subheader("📋 Dados Filtrados")
st.dataframe(df_filtrado)

st.caption("Fonte: Câmara de Comercialização de Energia Elétrica (CCEE)")
