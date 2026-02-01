import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Domain Imputer
class DomainImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_none = [
            'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'PoolQC', 'MiscFeature'
        ]
        self.cols_zero = ['GarageYrBlt', 'MasVnrArea']
        self.lot_frontage_map = {} 
        self.electrical_mode = None
        self.masvnr_mode = None

    def fit(self, X, y=None):
        X_temp = X.copy()
        self.lot_frontage_map = X_temp.groupby('Neighborhood')['LotFrontage'].median()
        self.electrical_mode = X_temp['Electrical'].mode()[0]
        self.masvnr_mode = X_temp['MasVnrType'].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_none:
            if col in X.columns: X[col] = X[col].fillna('None')
        
        if 'MasVnrType' in X.columns and 'MasVnrArea' in X.columns:
            mask_real_missing = X['MasVnrType'].isna() & (X['MasVnrArea'] > 0)
            X.loc[mask_real_missing, 'MasVnrType'] = self.masvnr_mode
            X['MasVnrType'] = X['MasVnrType'].fillna('None')

        for col in self.cols_zero:
            if col in X.columns: X[col] = X[col].fillna(0)
        
        if 'Electrical' in X.columns:
            X['Electrical'] = X['Electrical'].fillna(self.electrical_mode)

        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            neighborhood_medians = X['Neighborhood'].map(self.lot_frontage_map)
            X['LotFrontage'] = X['LotFrontage'].fillna(neighborhood_medians)
            X['LotFrontage'] = X['LotFrontage'].fillna(self.lot_frontage_map.median())
        return X

# 2. Feature Encoder (Ordinal Mapping)
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        self.qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                          'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        self.bsmt_exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
        self.bsmt_fin_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
        self.garage_fin_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
        self.functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 
                               'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
        self.fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}
        self.driveway_map = {'Y': 2, 'P': 1, 'N': 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.qual_cols:
            if col in X.columns: X[col] = X[col].map(self.quality_map)
        
        mappings = {
            'BsmtExposure': self.bsmt_exposure_map,
            'BsmtFinType1': self.bsmt_fin_map,
            'BsmtFinType2': self.bsmt_fin_map,
            'GarageFinish': self.garage_fin_map,
            'Functional': self.functional_map,
            'Fence': self.fence_map,
            'PavedDrive': self.driveway_map
        }
        for col, mapper in mappings.items():
            if col in X.columns: X[col] = X[col].map(mapper)
            
        if 'CentralAir' in X.columns: X['CentralAir'] = X['CentralAir'].map({'Y': 1, 'N': 0})
        if 'MSSubClass' in X.columns: X['MSSubClass'] = X['MSSubClass'].astype(str)
        return X

# 3. Feature Creator
class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TotalSF'] = (X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0))
        X['TotalBath'] = (X['FullBath'].fillna(0) + (0.5 * X['HalfBath'].fillna(0)) + 
                          X['BsmtFullBath'].fillna(0) + (0.5 * X['BsmtHalfBath'].fillna(0)))
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        X['HasRemodeled'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)
        X.loc[X['HouseAge'] < 0, 'HouseAge'] = 0
        X.loc[X['RemodAge'] < 0, 'RemodAge'] = 0
        return X

# 4. Skewness Transformer
class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold
        self.skewed_features = []

    def fit(self, X, y=None):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_features = X.select_dtypes(include=numerics).columns
        skew_features = X[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        self.skewed_features = skew_features[abs(skew_features) > self.skew_threshold].index
        self.skewed_features = [col for col in self.skewed_features if X[col].nunique() > 2]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.skewed_features] = np.log1p(X[self.skewed_features])
        return X