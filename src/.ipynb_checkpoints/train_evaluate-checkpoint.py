import pandas as pd
import yaml
import joblib
import json
import sys
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def get_model(model_name, params):
    """Obtiene una instancia del modelo basado en el nombre y los parámetros."""
    if model_name == 'linear_regression':
        return LinearRegression(**params)
    elif model_name == 'random_forest':
        return RandomForestRegressor(**params)
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

def train_evaluate(model_name, params, processed_data_path, model_output_path, metrics_output_path):
    """Entrena un modelo y guarda el modelo y las métricas."""
    
    # Crear directorios de salida
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos procesados
    X_train = pd.read_csv(Path(processed_data_path) / 'X_train.csv')
    y_train = pd.read_csv(Path(processed_data_path) / 'y_train.csv').iloc[:, 0]
    X_test = pd.read_csv(Path(processed_data_path) / 'X_test.csv')
    y_test = pd.read_csv(Path(processed_data_path) / 'y_test.csv').iloc[:, 0]
    
    # Obtener y entrenar el modelo
    model = get_model(model_name, params)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Guardar métricas [cite: 2054]
    metrics = {
        'rmse': rmse,
        'r2': r2
    }
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Guardar modelo
    joblib.dump(model, model_output_path)

if __name__ == "__main__":
    # Parser para recibir el nombre del modelo
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Nombre del modelo a entrenar (ej: random_forest)")
    args = parser.parse_args()

    # Cargar params.yaml
    try:
        with open('params.yaml', 'r') as f:
            all_params = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: params.yaml no encontrado.")
        sys.exit(1)
        
    model_params = all_params['train'][args.model]
    processed_data_dir = 'data/processed'
    model_out_file = f'models/{args.model}.joblib'
    metrics_out_file = f'metrics/metrics_{args.model}.json'
    
    train_evaluate(args.model, model_params, processed_data_dir, model_out_file, metrics_out_file)
    print(f"Modelo {args.model} entrenado. Métricas guardadas en {metrics_out_file}")