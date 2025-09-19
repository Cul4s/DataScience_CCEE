# =========================================
# Dashboard CCEE - Preços de Energia
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# -----------------------------
# 1️⃣ Carregar dados
# -----------------------------
df_raw = pd.read_csv("ccee_energia.csv", sep=";")

# "Derreter" as colunas de datas em linhas
df = df_raw.melt(id_vars=["Hora", "Submercado"],
                 var_name="Data",
                 value_name="PLD")

# Converter Data
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)

# Converter PLD para numérico
df["PLD"] = pd.to_numeric(df["PLD"], errors="coerce")

st.title("⚡ Dashboard CCEE - Preços de Energia")

# -----------------------------
# 2️⃣ Limpeza e tratamento
# -----------------------------
df.dropna(subset=["PLD"], inplace=True)
df.drop_duplicates(inplace=True)
df = df.sort_values("Data")

st.subheader("Visualização Inicial")
st.dataframe(df.head(10))

# -----------------------------
# 3️⃣ Filtros interativos
# -----------------------------
st.sidebar.header("Filtros do Dashboard")
submercados = df["Submercado"].unique()
selected_sub = st.sidebar.multiselect("Escolha Submercado(s)", submercados, submercados)
start_date = st.sidebar.date_input("Data Inicial", df["Data"].min())
end_date = st.sidebar.date_input("Data Final", df["Data"].max())

df_filtered = df[(df["Submercado"].isin(selected_sub)) &
                 (df["Data"] >= pd.to_datetime(start_date)) &
                 (df["Data"] <= pd.to_datetime(end_date))]

st.subheader(f"Dados Filtrados ({len(df_filtered)} registros)")
st.dataframe(df_filtered.head(10))

# -----------------------------
# 4️⃣ Modelo simples de previsão
# -----------------------------
df_filtered["Dia"] = (df_filtered["Data"] - df_filtered["Data"].min()).dt.days
X = df_filtered[["Dia"]]
y = df_filtered["PLD"]

if len(df_filtered) > 10:  # só roda previsão se tiver dados suficientes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("📈 Regressão Linear - Previsão do PLD")
    st.write(f"RMSE: {rmse:.2f} R$/MWh")

    # -----------------------------
    # 5️⃣ Gráficos
    # -----------------------------
    st.subheader("Evolução do PLD ao longo do tempo")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df_filtered, x="Data", y="PLD", hue="Submercado", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Distribuição do PLD")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_filtered["PLD"], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # 6️⃣ Previsão em tempo real
    # -----------------------------
    st.subheader("🔮 Prever PLD em função da Data")
    future_days = st.number_input("Quantos dias após a última data da base?", min_value=1, value=10)
    last_day = (df_filtered["Data"].max() - df_filtered["Data"].min()).days
    predicted_pld = model_reg.predict([[last_day + future_days]])[0]
    st.success(f"💰 PLD Previsto em {future_days} dias: R$ {predicted_pld:.2f} / MWh")
else:
    st.warning("Poucos dados filtrados para rodar previsão.")
