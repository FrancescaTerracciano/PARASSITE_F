import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
    st.title("Dashboard illustrativa: Parassiti Maschili sotto analisi 🔍🪳")
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
    st.subheader("Analisi di Temperatura e Umidità 🌡️")
    st.write("Questo grafico mostra l'andamento della temperatura e dell'umidità nel periodo selezionato.")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['temperature_mean'], label='Temperatura', ax=ax)
    sns.lineplot(x=filtered_data['Date'], y=filtered_data['relativehumidity_mean'], label='Umidità', ax=ax, color='red')
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

    # Box Plot per Temperatura e Umidità
    st.subheader("Box Plot per Temperatura e Umidità")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=filtered_data[['temperature_mean', 'relativehumidity_mean']], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Scatter Plot con Regressione
    st.subheader("Relazione tra Temperatura e Adulti Maschi")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.regplot(x=filtered_data['temperature_mean'], y=filtered_data['no. of Adult males'], ax=ax)
    st.pyplot(fig)

    # Istogramma per il Numero di Adulti Maschi
    st.subheader("Distribuzione del Numero di Adulti Maschi")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_data['no. of Adult males'], bins=20, kde=True, color='purple', ax=ax)
    st.pyplot(fig)

    # Grafico a Torta per il Numero di Adulti Maschi
    st.subheader("Distribuzione Percentuale del Numero di Adulti Maschi")
    fig, ax = plt.subplots()
    filtered_data['Category'] = pd.cut(filtered_data['no. of Adult males'], bins=[0, 10, 20, 30, 40, 50], labels=['0-10', '10-20', '20-30', '30-40', '40-50'])
    filtered_data['Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Sezione di previsione
    st.sidebar.subheader("Effettua la predizione di Adulti Maschi 📊")
    temp_input = st.sidebar.number_input("Inserisci la Temperatura Media", value=float(data['temperature_mean'].mean()))
    humidity_input = st.sidebar.number_input("Inserisci l'Umidità Relativa Media", value=float(data['relativehumidity_mean'].mean()))
    if st.sidebar.button("Prevedi"):
        prediction = model.predict([[temp_input, humidity_input]])[0]
        st.sidebar.write(f"Il modello ha predetto come numero di adulti maschi per i dati forniti: {prediction:.2f}")

if __name__ == "__main__":
    main()
