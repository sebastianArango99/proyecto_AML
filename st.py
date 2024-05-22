import streamlit as st
import pandas as pd
import pickle

def load_model(pickle_file):
    with open(pickle_file, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(df):
    df = df.fillna(0)
    return df

def main():
    st.title("FARMU - Predicción de Cajas Para Equipo de Última Milla")
    
    st.markdown("### Descargar plantilla CSV")
    with open("template.csv", "rb") as file:
        btn = st.download_button(
            label="Descargar plantilla",
            data=file,
            file_name="template.csv",
            mime="text/csv"
        )
    
    csv_file = st.file_uploader("Ingrese un archivo en formato CSV", type="csv")
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Muestra del archivo:")
        st.write(df.head())
        
        model_file = 'path/to/your/random_forest_model.pkl'  # Update with the correct path
        model = load_model(model_file)
        
        df_processed = preprocess_data(df)
        
        predictions = model.predict(df_processed)
        
        st.write("Predicciones:")
        st.write(predictions)

if __name__ == "__main__":
    main()