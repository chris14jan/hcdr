import pandas as pd
from hcdr.data.data import Data
from hcdr.data.tbl_preproc import preprocess_credit_card_balance_df, get_final_bureau_merged, preprocess_installments_payments_df, preprocess_POS_CASH_balance_df

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