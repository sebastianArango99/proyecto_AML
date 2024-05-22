import streamlit as st
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def preprocess_data(df):
    df = df.fillna(0)
    return df

def main():
    st.title("FARMU - Predicción de Cajas Para Equipo de Última Milla")
    
    st.markdown("### Descargar plantilla CSV")
    with open("template2 - Sheet1.csv", "rb") as file:
        btn = st.download_button(
            label="Descargar plantilla",
            data=file,
            file_name="template2 - Sheet1.csv",
            mime="text/csv"
        )
    
    csv_file = st.file_uploader("Ingrese un archivo en formato CSV", type="csv")
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Muestra del archivo:")
        st.write(df.head())
        
        model_file = 'best_model_oxes.pkl'  # Update with the correct path
        model = load_model(model_file)
        
        df_processed = preprocess_data(df)
        
        predictions = model.predict(df_processed)
        
        st.write("Predicciones:")
        st.write(predictions)

        total_boxes = sum(predictions)
        
        
        st.write(f"Total de cajas: {total_boxes}")
        
        # Plotting the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(predictions, marker='o')
        plt.title("Cantidad de pedidos por cada predicción")
        plt.xlabel("Índice de Predicción")
        plt.ylabel("Cantidad de Cajas")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
