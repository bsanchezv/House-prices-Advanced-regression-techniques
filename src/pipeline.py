from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector

# Importamos las clases que acabamos de guardar en transformers.py
from src.transformers import DomainImputer, FeatureEncoder, FeatureCreator, SkewnessTransformer

def get_training_pipeline(model_instance):
    """
    Crea un pipeline completo que incluye:
    1. Limpieza y Feature Engineering Personalizado.
    2. OneHotEncoding para categóricas y StandardScaling para numéricas.
    3. El modelo final (Lasso, Ridge, etc.).
    """
    
    # 1. Pipeline de Ingeniería de Características (Secuencial)
    feature_engineering = Pipeline([
        ('imputer', DomainImputer()),
        ('encoder', FeatureEncoder()),
        ('creator', FeatureCreator()),
        ('skewness', SkewnessTransformer())
    ])

    # 2. Preprocesamiento Final (Dividir Numéricas vs Nominales)
    # Las columnas "object" que queden son las Nominales (Barrios, etc.) -> OneHot
    # El resto son Numéricas (incluyendo las Ordinales que ya convertimos a números) -> StandardScale
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), make_column_selector(dtype_exclude='object')),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include='object'))
        ],
        verbose_feature_names_out=False
    )

    # 3. Pipeline Final: Ingeniería -> Preprocesamiento -> Modelo
    final_pipeline = Pipeline([
        ('feature_eng', feature_engineering),
        ('preprocessing', preprocessor),
        ('model', model_instance)
    ])
    
    return final_pipeline