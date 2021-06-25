import pandas as pd
from hcdr.data.data import Data
from hcdr.data.tbl_preproc import preprocess_credit_card_balance_df, get_final_bureau_merged, preprocess_installments_payments_df, preprocess_POS_CASH_balance_df
import os
from pathlib import Path

### Aggregated dfs ###
def agg_dfs():
    print("Aggregating non-application dataframes...")
    
    # Placeholder for all aggregated dfs:
    df_dict = Data().get_data(tables=["application_train", "application_test", "previous_application"])
    df_dict = Data().drop_missing_cols_dict(df_dict, missing_amt=0.3, verbose=True)
    
    # Customized aggregated dfs:
    df_dict["credit_card_balance"] = Data().df_optimized(preprocess_credit_card_balance_df())
    df_dict["bureau"] = Data().df_optimized(get_final_bureau_merged())
    df_dict["installments_payments"] = Data().df_optimized(preprocess_installments_payments_df())
    df_dict["POS_CASH_balance"] = Data().df_optimized(preprocess_POS_CASH_balance_df())
    
    df_dict = Data().drop_missing_cols_dict(df_dict, missing_amt=0.3, verbose=True)
    
    dfs_dict = df_dict.copy()
    
    # Tables that do not require aggregration:
    dfs_dict.pop('application_train', None)
    dfs_dict.pop('application_test', None)
    dfs_dict.pop('credit_card_balance', None) # Using Max's function.
    dfs_dict.pop('installments_payments', None) # Using Max's function.
    dfs_dict.pop('POS_CASH_balance', None) # Using Max's function.
    dfs_dict.pop('bureau', None) # Using Keji's function.
    
    # Temporary aggregation loop to test pipeline downstream:
    for df_key, df in dfs_dict.items():
        df_dict[df_key] = df.groupby(by="SK_ID_CURR").agg("mean")
    return df_dict

# Merge aggregated dfs
def merge_dfs(df_app="application_train", verbose=True):
    """ """
    
    df_dict = agg_dfs()
    dfs_dict = df_dict.copy()
    df_merged = df_dict[df_app]
    try: df_merged = df_merged.drop(columns="SK_ID_PREV")
    except: pass
    
    dfs_dict.pop('application_train', None)
    dfs_dict.pop('application_test', None)
    dfs_dict.pop('bureau_balance', None)
    
    for df_key, df in dfs_dict.items():
        try: df = df.drop(columns="SK_ID_PREV")
        except: pass
        
        if verbose:
            print(f"Merging {df_key} table ...")
        
        df_merged = pd.merge(df_merged, df, on="SK_ID_CURR", how="left")
        
        if verbose:
            print(f"Original shape: {df_dict[df_app].shape[0]}, shape after merge: {df_merged.shape[0]}")
    return df_merged

# Model Data
def training_data(load_saved=False):
    """
    Return (X_train, y_train)"""
    
    root_dir = Path(__file__).parents[0]
    df_merged_path = os.path.join(root_dir, f"merged_tables/df_merged_train.pkl")
    
    if load_saved:
        df_merged = pd.read_pickle(df_merged_path)
    else:
        print("CREATING MERGED DF app_train...")
        df_merged = merge_dfs(df_app="application_train", verbose=True)
        print("SAVING MERGED DF app_train...")
        df_merged.to_pickle(df_merged_path)
    
    print("SPLITTING MERGED DF INTO X and Y...")
    X_train = df_merged.drop(columns=["SK_ID_CURR", "TARGET"])
    y_train = df_merged["TARGET"]
    
    print("DELETING MERGED DF FROM MEMORY...")
    del df_merged
    
    return (X_train, y_train)

def test_data(load_saved=False):
    """
    Return X_test"""
    
    root_dir = Path(__file__).parents[0]
    df_merged_path = os.path.join(root_dir, f"merged_tables/df_merged_test.pkl")
    
    if load_saved:
        df_merged = pd.read_pickle(df_merged_path)
    else:
        print("CREATING MERGED DF app_test...")
        df_merged = merge_dfs(df_app="application_test", verbose=True)
        print("SAVING MERGED DF app_train...")
        df_merged.to_pickle(df_merged_path)
        
    X_test = df_merged.drop(columns=["SK_ID_CURR"])
    
    print("DELETING MERGED DF FROM MEMORY...")
    del df_merged
    
    return X_test

if __name__ == "__main__": 
    training_data()