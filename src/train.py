import sys
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Ignoramos warnings para mantener la consola limpia
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from data_loader import load_and_split_data
except ImportError:
    # Fallback por si acaso se ejecuta desde dentro de src (no recomendado)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_loader import load_and_split_data

# Importamos el pipeline que SÍ está dentro de src
from src.pipeline import get_training_pipeline

def load_config():
    # Asumimos que config.yaml está en la raíz del proyecto
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_training():
    print("🚀 Iniciando proceso de entrenamiento...")
    config = load_config()
    
    # 1. Cargar Datos
    print("📂 Cargando datos...")
    # data_loader ya se encarga de usar los paths correctos definidos en config
    X_train, X_val, y_train, y_val = load_and_split_data(config)
    
    # 2. Transformar Target (Logaritmo)
    # Importante: Usamos log1p para evitar errores con log(0) si hubiera precios 0
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    # 3. Definir Experimentos
    experiments = [
        {
            "name": "Regresión Lineal (Base)",
            "model": LinearRegression(),
            "params": {} 
        },
        {
            "name": "Lasso (L1)",
            "model": Lasso(max_iter=10000), # Aumentamos iteraciones para evitar warnings de convergencia
            "params": {'model__alpha': config['hyperparameters']['LASSO_ALPHAS']}
        },
        {
            "name": "Ridge (L2)",
            "model": Ridge(),
            "params": {'model__alpha': config['hyperparameters']['RIDGE_ALPHAS']}
        }
    ]
    
    results = []

    # 4. Bucle de Entrenamiento
    for exp in experiments:
        print(f"\n⚡ Entrenando: {exp['name']}...")
        
        pipeline = get_training_pipeline(exp['model'])
        
        grid = GridSearchCV(
            pipeline, 
            param_grid=exp['params'], 
            cv=5, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train_log)
        
        # 5. Evaluación
        best_model = grid.best_estimator_
        y_pred_log = best_model.predict(X_val)
        
        # Revertir logaritmo (expm1 es la inversa de log1p)
        y_pred_real = np.expm1(y_pred_log)
        y_val_real = np.expm1(y_val_log)
        
        rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
        r2 = r2_score(y_val_real, y_pred_real)
        
        print(f"   ✅ Mejor Params: {grid.best_params_}")
        print(f"   📊 RMSE: ${rmse:,.2f}")
        print(f"   📈 R2: {r2:.4f}")
        
        results.append({
            "Modelo": exp['name'],
            "RMSE ($)": round(rmse, 2),
            "R2": round(r2, 4),
            "Best Params": str(grid.best_params_)
        })

    # 6. Resumen Final
    print("\n" + "="*40)
    print("🏆 TABLA DE RESULTADOS FINAL")
    print("="*40)
    results_df = pd.DataFrame(results)
    print(results_df)
    
    results_df.to_csv("resultados_finales.csv", index=False)
    print("\n✅ Resultados guardados en 'resultados_finales.csv'")

if __name__ == "__main__":
    run_training()