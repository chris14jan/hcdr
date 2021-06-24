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

    def df_optimized(self, df, verbose=True, **kwargs):
        """
        Reduces size of dataframe by downcasting numeircal columns
        :param df: input dataframe
        :param verbose: print size reduction if set to True
        :param kwargs:
        :return: df optimized
        """
        in_size = df.memory_usage(index=True).sum()
        # Optimized size here
        for _type in ["float", "integer"]:
            l_cols = list(df.select_dtypes(include=_type))
            for col in l_cols:
                df[col] = pd.to_numeric(df[col], downcast=_type)
                if _type == "float":
                    df[col] = pd.to_numeric(df[col], downcast="integer")
        out_size = df.memory_usage(index=True).sum()
        ratio = (1 - round(out_size / in_size, 2)) * 100
        GB = out_size / 1000000000
        if verbose:
            print("optimized size by {} % | {} GB".format(ratio, GB))
        return df
    
    def get_data(self, root_dir = Path(__file__).parents[2], tables=["application_train", "application_test", "bureau", "bureau_balance", "POS_CASH_balance", 
                               "credit_card_balance", "previous_application", "installments_payments"], nrows=None):
        
        dict_dict_dtypes = self.map_dtype_cols()

        df_dict = {}
        for table in tables: 
            data_dir_path = os.path.join(root_dir, f"raw_data/{table}" + ".csv")
            df_temp = pd.read_csv(data_dir_path, dtype=dict_dict_dtypes[table], nrows=nrows)
            df_dict[table] = self.df_optimized(df_temp, verbose=False)
        return df_dict

    def drop_missing_cols_df(self, df, missing_amt=0.3, verbose=True):
        """
        Pass a df. 
        Returns df without columns containing missing data >= missing_amt
        missing_amt : float 0.0 - 1.0
        """
        df_missing_vals=pd.DataFrame((df.isnull().sum()/df.shape[0]).sort_values(ascending=False)).reset_index().rename(columns={"index":"feature", 0:"percent_missing"})
        drop_cols = list(df_missing_vals[df_missing_vals.percent_missing >= missing_amt].feature.values)#+["FLAG_MOBIL", "FLAG_WORK_PHONE"]
        df_reduced = df.drop(columns=drop_cols)
        if verbose:
            print(f"{df.shape[1]} --> {df_reduced.shape[1]}. Dropped {df.shape[1] - df_reduced.shape[1]} columns")
        return df_reduced

    def drop_missing_cols_dict(self, df_dict, missing_amt=0.3, verbose=True):
        """
        Pass a dict of dfs. 
        Returns dict of dfs without columns containing missing data >= missing_amt
        missing_amt : float 0.0 - 1.0
        """
        dfs_dropped_missing_vals = {}
        dict_dropped_cols = {}
        for df_keys, df in df_dict.items():
            df_missing_vals=pd.DataFrame((df.isnull().sum()/df.shape[0]).sort_values(ascending=False)).reset_index().rename(columns={"index":"feature", 0:"percent_missing"})
            drop_cols = list(df_missing_vals[df_missing_vals.percent_missing >= missing_amt].feature.values)#+["FLAG_MOBIL", "FLAG_WORK_PHONE"]
            dict_dropped_cols[df_keys]=drop_cols
            df_reduced = df.drop(columns=drop_cols)
            if verbose:
                print(f"{df_keys}:       {df.shape[1]} --> {df_reduced.shape[1]}. Dropped {df.shape[1] - df_reduced.shape[1]} columns")
            dfs_dropped_missing_vals[df_keys] = df_reduced
        return dfs_dropped_missing_vals


if __name__ == "__main__": 
    data = Data()
    print(data.get_data())