import pandas as pd
from hcdr.data.data import Data
from hcdr.data.cc_bal import preprocess_credit_card_balance_df

### Aggregated dfs ### ########################################### remove nrows kwarg ###########################################
def agg_dfs():
    print("Aggregating non-application dataframes...")
    
    # Placeholder for all aggregated dfs:
    ### TODO ###
    df_dict = Data().get_data(nrows=100_000) ########################################### remove nrows kwarg ###########################################
    ### TODO ###
    df_dict = Data().drop_missing_cols_dict(df_dict, missing_amt=0.3, verbose=True)
    
    # Update placeholder with customized aggregated dfs:
    df_cc_agged = Data().df_optimized(preprocess_credit_card_balance_df())
    df_dict["credit_card_balance"] = df_cc_agged
    df_dict = Data().drop_missing_cols_dict(df_dict, missing_amt=0.3, verbose=True)
    
    dfs_dict = df_dict.copy()
    # Tables that do not require aggregration:
    dfs_dict.pop('application_train', None)
    dfs_dict.pop('application_test', None)
    dfs_dict.pop('bureau_balance', None) # No SK_ID_CURR. Will be aggregated separately with 'bureau'.
    dfs_dict.pop('credit_card_balance', None) # Using Max's function.
    
    # Temporary aggregation loop to test pipeline downstream:
    for df_key, df in dfs_dict.items():
        df_dict[df_key] = df.groupby(by="SK_ID_CURR").agg("mean")
    return df_dict

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