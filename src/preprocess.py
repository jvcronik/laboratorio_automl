import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import yaml
import sys

def preprocess_data(raw_data_path, processed_data_path, params):
    """
    Carga los datos, los divide y escala.
    """
    # Crear directorios de salida si no existen
    Path(processed_data_path).mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    df = pd.read_csv(raw_data_path)
    
    # Definir características (X) y objetivo (y)
    X = df.drop(params['target_column'], axis=1)
    y = df[params['target_column']]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_state']
    )
    
    # Escalar características (solo numéricas)
    # En este dataset, todas son numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de nuevo a DataFrames para guardar con columnas
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Guardar datos procesados
    X_train_df.to_csv(Path(processed_data_path) / 'X_train.csv', index=False)
    X_test_df.to_csv(Path(processed_data_path) / 'X_test.csv', index=False)
    y_train.to_csv(Path(processed_data_path) / 'y_train.csv', index=False)
    y_test.to_csv(Path(processed_data_path) / 'y_test.csv', index=False)

if __name__ == "__main__":
    # Cargar params.yaml
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: params.yaml no encontrado.")
        sys.exit(1)
        
    raw_path = 'data/dataset_v1.csv'
    processed_path = 'data/processed'
    
    preprocess_data(raw_path, processed_path, params['preprocess'])
    print(f"Datos procesados y guardados en {processed_path}")