import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#import numpy as np

# Funzione per caricare i dati
@st.cache_resource
def load_data():
    data = pd.read_excel("temp_humid_data.xlsx", sheet_name='Sheet3')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Funzione per addestrare il modello di regressione
def train_model(data):
    X = data[['temperature_mean', 'relativehumidity_mean']]
    y = data['no. of Adult males']
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    st.title("Dashboard di Analisi e Predizione per Sheet3 - temp_humid_data")
    data = load_data()
    model = train_model(data)

    # Sidebar per filtri
    st.sidebar.subheader("Filtri Visualizzazione")
    start_date = st.sidebar.date_input("Data Iniziale", data['Date'].min().date())
    end_date = st.sidebar.date_input("Data Finale", data['Date'].max().date())

    # Filtra i dati in base all'intervallo di date selezionato
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Grafico lineare per temperatura e umidità
    st.subheader("Analisi di Temperatura e Umidità")
    st.write("Questo grafico mostra l'andamento della temperatura e dell'umidità nel periodo selezionato.")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['temperature_mean'], label='Temperatura', ax=ax)
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['relativehumidity_mean'], label='Umidità', ax=ax, color='orange')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    # Grafico a barre per il numero di adulti maschi
    st.subheader("Trend del Numero di Adulti Maschi")
    st.write("Visualizzazione del numero di adulti maschi nel periodo selezionato.")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_data = filtered_data.iloc[::5, :]  # Mostra solo ogni quinta data
    sns.barplot(x=plot_data['Date'].dt.strftime('%Y-%m-%d'), y=plot_data['no. of Adult males'], ax=ax, color='blue')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Statistiche descrittive
    st.subheader("Statistiche Descrittive")
    st.write("Statistiche descrittive delle variabili nel dataset.")
    st.write(filtered_data.describe())

    # Mappa di calore per le correlazioni
    st.subheader("Mappa di Calore delle Correlazioni")
    st.write("Questa mappa di calore mostra le correlazioni tra temperatura, umidità e numero di adulti maschi.")
    corr = filtered_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # Sezione di previsione
    st.sidebar.subheader("Predizione di Adulti Maschi")
    temp_input = st.sidebar.number_input("Inserisci la Temperatura Media", value=float(data['temperature_mean'].mean()))
    humidity_input = st.sidebar.number_input("Inserisci l'Umidità Relativa Media", value=float(data['relativehumidity_mean'].mean()))
    if st.sidebar.button("Prevedi"):
        prediction = model.predict([[temp_input, humidity_input]])[0]
        st.sidebar.write(f"Numero previsto di adulti maschi: {prediction:.2f}")

if __name__ == "__main__":
    main()
