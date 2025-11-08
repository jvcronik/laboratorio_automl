import json
import pandas as pd
from pathlib import Path
import yaml
import sys

def select_best_model(metrics_path, report_file, params):
    """Carga todas las métricas, selecciona el mejor modelo y genera un reporte."""

    metrics_files = list(Path(metrics_path).glob('*.json'))
    if not metrics_files:
        print("No se encontraron archivos de métricas.")
        return

    metric_to_optimize = params['metric_to_optimize']
    # Los modelos de regresión buscan minimizar el RMSE
    lower_is_better = True if metric_to_optimize == 'rmse' else False

    results = []
    for mfile in metrics_files:
        model_name = mfile.stem.replace('metrics_', '')
        with open(mfile, 'r') as f:
            metrics = json.load(f)
        metrics['model'] = model_name
        results.append(metrics)

    if not results:
        print("La lista de resultados está vacía.")
        return

    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=metric_to_optimize, ascending=lower_is_better)

    best_model = results_df.iloc[0]

    # Generar reporte
    with open(report_file, 'w') as f:
        f.write("# Reporte de Comparación de Modelos (AutoML)\n\n")
        f.write("Resultados de todos los experimentos:\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## Mejor Modelo\n\n")
        f.write(f"El mejor modelo encontrado fue **{best_model['model']}**.\n")
        f.write(f"Métrica ({metric_to_optimize}): **{best_model[metric_to_optimize]:.4f}**\n")

if __name__ == "__main__":
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: params.yaml no encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer params.yaml: {e}") # Ayuda para debuggear
        sys.exit(1)

    metrics_dir = 'metrics'
    report_path = 'report.md'

    select_best_model(metrics_dir, report_path, params['select_best'])
    print(f"Reporte de selección de modelo guardado en {report_path}")