#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:05:44 2024

@author: fcdn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.impute import KNNImputer

from scipy.stats import yeojohnson, yeojohnson_normmax


from scipy.stats import yeojohnson
# from scipy.special import inv_boxcox

def remove_constant_cols(X):
    
    unique_counts = X.nunique()
    # Filter columns with only one unique value (constant columns)
    constant_columns = unique_counts[unique_counts == 1].index
    # Print or use the constant columns
    print("Columns with constant values:", constant_columns)
    X = X.drop(columns=constant_columns)
    return X

def fix_NaNs(X):    
    # %% Fix NaNs in training data
    columns_with_nan = X.columns[X.isna().any()].tolist()
    # 'columns_with_nan' will contain a list of column names with NaN values
    print("Co0lumns with NaN values:", columns_with_nan)
    
    imputer = KNNImputer(n_neighbors=5)
    
    for covariate in columns_with_nan:
        print(covariate)  
        X[covariate] = imputer.fit_transform(X[covariate].values.reshape(-1, 1))
    
    # Final Check
    X.isnull().values.any()
    return X

# def yeo(X):
#     X_t, lmbda = yeojohnson(X+1)  
#     return X_t, lmbda
 
def yeo(X):
    df = pd.DataFrame(X)
    lmbda_values = df.apply(lambda x: yeojohnson_normmax(x))
    df_transformed = df.apply(lambda x: yeojohnson(x + 1, lmbda=lmbda_values[x.name]))
    X_t = np.array(df_transformed)
    return X_t, lmbda_values

def yeo_lmbda(X,lmbda_values):
    df = pd.DataFrame(X)
    df_transformed = df.apply(lambda x: yeojohnson(x + 1, lmbda=lmbda_values[x.name]))
    X_t = np.array(df_transformed)
    return X_t