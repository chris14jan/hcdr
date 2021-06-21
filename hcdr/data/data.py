import pandas as pd
import os 
from pathlib import Path

class Data: 
    def __init__(self):
        pass

    def get_small_data(self, tables=["application_train", "application_test", "bureau", "bureau_balance", "POS_CASH_balance", 
                               "credit_card_balance", "previous_application", "installments_payments"], 
                       nrows=1_000):
        """Returns dict of dfs with the first # of rows specified"""
        root_dir = Path(__file__).parents[2]
        df_dict = {}
        
        for table in tables: 
            data_dir_path = os.path.join(root_dir, f"raw_data/{table}" + ".csv")
            df_dict[table] =  pd.read_csv(data_dir_path, nrows=nrows)
        return df_dict

    def map_dtype_cols(self):
        """Returns a dict of dicts mapping each df's numerical columns to float32 or int32 dtypes.
        This reduces the entire data load time to 1/3 of the original load time (1 min instead of 3 min)"""
        dict_of_dfs = self.get_small_data()
        dict_dict_dtypes = {}
        for df_key, df in dict_of_dfs.items():
            df_temp = df.copy()
            # dict -> col_name:col_dtype for float64 to float32 dtypes
            col_dtypes = {col:'float32' for col in df_temp.select_dtypes(include='float64').columns.values}
            # update dict -> col_name:col_dtype for int64 to int32 dtypes
            for col in df_temp.select_dtypes(include='int64').columns.values:
                col_dtypes[col] = 'int32'
            # save dtypes map dict
            dict_dict_dtypes[df_key] = col_dtypes
        return dict_dict_dtypes

    def get_data(self, tables=["application_train", "application_test", "bureau", "bureau_balance", "POS_CASH_balance", 
                               "credit_card_balance", "previous_application", "installments_payments"], nrows=None):
        
        root_dir = Path(__file__).parents[2]
        
        dict_dict_dtypes = self.map_dtype_cols()
                
        df_dict = {}
        for table in tables: 
            data_dir_path = os.path.join(root_dir, f"raw_data/{table}" + ".csv")
            df_dict[table] =  pd.read_csv(data_dir_path, dtype=dict_dict_dtypes[table], nrows=nrows)
        
        return df_dict


if __name__ == "__main__": 
    data = Data()
    print(data.get_data())