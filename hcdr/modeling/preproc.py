from timeit import default_timer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np

from hcdr.data.merged_data import merge_dfs

def scaler_func(scaler_type="standard"):
    """ """
    scaler_dict = {"standard" : StandardScaler(),
                   "robust"   : RobustScaler(),
                   "minmax"   : MinMaxScaler()}
    return scaler_dict[scaler_type]

def preproc_pipeline(scaler_type=None, verbose=True):
    """
    scaler_type : str {"standard" : StandardScaler(),
                       "robust"   : RobustScaler(),
                       "minmax"   : MinMaxScaler()}
                : None returns preproc pipeline without scaler
       """
    
    if verbose:
        print("Running preprocessor...")
        print(f"scaler_type={scaler_type}")
    
    # Encode categorical variables
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="missing")),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    num_transformer = Pipeline([
       ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ])

    # Paralellize "num_transformer" and "One hot encoder"
    preprocessor1 = ColumnTransformer([
    # TODO # explicitly list columns
        ('num_transformer', num_transformer, selector(dtype_exclude=object)),
        ('cat_transformer', cat_transformer, selector(dtype_include=object))
    ])

    if scaler_type == None:
        return preprocessor1
    else:
        preprocessor2 = Pipeline([
            ('preprocessor1', preprocessor1),
            ('scaler', scaler_func(scaler_type=scaler_type))
        ])    
        return preprocessor2
    
if __name__ == "__main__": 
    
    df_merged = merge_dfs(df_app="application_train", verbose=True).iloc[:1000]
    X = df_merged.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df_merged["TARGET"]
    
    preproc = preproc_pipeline(scaler_type=None)
    X_transformed = preproc.fit(X)