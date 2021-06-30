from hcdr.data.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def preprocess_credit_card_balance_df(agg="mean"):
    """ A function that retrieves, preprocesses, feature engineers and groups the credit_card_balance_df by SK_ID_CURR using the mean """

    print("Loading credit card balance data table...")
    credit_card_balance_df = Data().get_data(tables=["credit_card_balance"])["credit_card_balance"]
    credit_card_balance_df = Data().drop_missing_cols_df(credit_card_balance_df, missing_amt=0.3, verbose=True)
    credit_card_balance_df["Credit Utilization Ratio"] = credit_card_balance_df["AMT_BALANCE"] / credit_card_balance_df["AMT_CREDIT_LIMIT_ACTUAL"]

    labels = ["0.00 - 0.25", "0.25 - 0.50", "0.50 - 0.75", "0.75 - 1.00", "> 1."]
    bins = [0, .25, .5, .75, 1., 1000]
    credit_card_balance_df["Credit Utilization Ratio"] = pd.cut(credit_card_balance_df["Credit Utilization Ratio"], bins=bins, labels=labels).fillna("0.00 - 0.25")

    ohe = OneHotEncoder(sparse = False)
    ohe.fit(credit_card_balance_df[["Credit Utilization Ratio"]])
    ratio_encoded = ohe.transform(credit_card_balance_df[["Credit Utilization Ratio"]])
    credit_card_balance_df[labels[0]], credit_card_balance_df[labels[1]], credit_card_balance_df[labels[2]], credit_card_balance_df[labels[3]], credit_card_balance_df[labels[4]] = ratio_encoded.T

    columns_to_drop = ["MONTHS_BALANCE", "AMT_BALANCE", "Credit Utilization Ratio", "AMT_CREDIT_LIMIT_ACTUAL"]
    credit_card_balance_df = credit_card_balance_df.drop(columns=columns_to_drop)

    preprocessed_df = credit_card_balance_df.groupby("SK_ID_CURR").agg(agg)

    return preprocessed_df

def preprocess_installments_payments_df(agg="mean"):
    """ A function that retrieves and groups the installments_payments_df by SK_ID_CURR using the mean """
    
    print("Loading installments payments data table...")
    installments_payments_df = Data().get_data(tables=["installments_payments"])["installments_payments"]
    installments_payments_df = Data().drop_missing_cols_df(installments_payments_df, missing_amt=0.3, verbose=True)

    columns_to_drop = ["NUM_INSTALMENT_VERSION"]
    installments_payments_df = installments_payments_df.drop(columns=columns_to_drop)

    grouped_df = installments_payments_df.groupby("SK_ID_CURR").agg(agg)

    return grouped_df

def preprocess_POS_CASH_balance_df(agg="mean"):

    """ A function that retrieves and groups the POS_CASH_balance_df by SK_ID_CURR using the mean"""

    print("Loading POS_CASH_balance data table...")
    POS_CASH_balance_df = Data().get_data(tables=["POS_CASH_balance"])["POS_CASH_balance"]
    POS_CASH_balance_df = Data().drop_missing_cols_df(POS_CASH_balance_df, missing_amt=0.3, verbose=True)

    columns_to_drop = []
    POS_CASH_balance_df = POS_CASH_balance_df.drop(columns= columns_to_drop)

    grouped_df = POS_CASH_balance_df.groupby("SK_ID_CURR").agg(agg)

    return grouped_df


# Bureau
def debt_credit_ratio(df):
    """Calculate debt to credit ratio. Pass bureau df"""

    bureau = df

    bureau['debt_credit_ratio_limit']=bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']

    bureau['debt_credit_ratio_sum']=bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']

    return bureau

def percent_overdue(df):
    """Calculate percent overdue"""

    bureau = debt_credit_ratio(df)

    bureau['overdue_percent_limit'] = bureau[
        'AMT_CREDIT_MAX_OVERDUE'] / bureau['AMT_CREDIT_SUM_LIMIT']

    bureau['overdue_percent_sum'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau[
        'AMT_CREDIT_SUM']

    return bureau

def encode_credit_active(df):
    """Encode Credit_Active Column"""

    bureau = percent_overdue(df)

    ohe = OneHotEncoder(sparse=False)

    credit_act = bureau[['CREDIT_ACTIVE']]

    credit_act_OHE = ohe.fit_transform(credit_act)

    f = ohe.get_feature_names()

    bureau[f[0]], bureau[f[1]], bureau[f[2]], bureau[f[3]] = credit_act_OHE.T

    return bureau

def encode_credit_type(df):
    """Encode Credit_Type Column"""

    bureau = encode_credit_active(df)

    credit_type = pd.get_dummies(bureau['CREDIT_TYPE'])
    credit_type['SK_ID_BUREAU'] = bureau['SK_ID_BUREAU']
    bureau = bureau.merge(credit_type, how='inner', on='SK_ID_BUREAU')

    return bureau

def get_bureau_final(df):
    """Final bureau df with encoded columns and new features"""

    bureau = encode_credit_type(df)

    bureau_drop = [
        'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'CREDIT_ACTIVE',
        'CREDIT_TYPE'
    ]

    bureau = bureau.drop(columns=bureau_drop)

    return bureau

def get_bureau_balance_final(df):
    """Final bureau_balance df"""

    bureau_balance = df

    bureau_balance_status = pd.get_dummies(bureau_balance['STATUS'])
    bureau_balance_status['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU']
    bureau_balance = pd.concat(
        [bureau_balance,
         bureau_balance_status.drop('SK_ID_BUREAU', axis=1)],
        axis=1)

    bureau_balance = bureau_balance.drop(columns=['STATUS'])
    bureau_bal_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(
        ['count', 'mean', 'max', 'min', 'std', 'nunique'])
    bureau_bal_agg.columns = bureau_bal_agg.columns.to_flat_index().str.join(
        '_')

    return bureau_bal_agg

def get_final_bureau_merged():
    """This function calls all previous functions to get the final merged df"""

    print("Loading 'bureau' AND 'bureau_balance' data tables...")
    df_dict = Data().get_data(tables=['bureau', 'bureau_balance'])
    # df_dict = Data().drop_missing_cols_dict(df_dict, missing_amt=0.3, verbose=True)

    df = df_dict['bureau']
    df_bal = df_dict['bureau_balance']

    print("Running get_bureau_final function...")
    bureau = get_bureau_final(df)
    print("Running get_bureau_balance_final function...")
    bureau_bal_agg = get_bureau_balance_final(df_bal)

    bureau_merged = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU')
    bureau_merged = bureau_merged.drop(columns=['CREDIT_CURRENCY'])

    # agg SK_ID_BUREAU column
    bureau_count = bureau_merged[['SK_ID_CURR', 'SK_ID_BUREAU']]
    bureau_count = bureau_count.groupby('SK_ID_CURR').agg(['count'])

    # flatten multi-index column names
    bureau_count.columns = bureau_count.columns.to_flat_index().str.join('_')

    # merge into df
    bureau_merged_f = bureau_merged.merge(bureau_count, on='SK_ID_CURR')
    bureau_merged_f = bureau_merged_f.drop(columns=['SK_ID_BUREAU'])

    # agg df
    bureau_merged_agg = bureau_merged_f.groupby('SK_ID_CURR').agg(
        ['count', 'mean', 'max', 'min', 'std', 'nunique'])

    # flatten multi-index columns
    bureau_merged_agg.columns = bureau_merged_agg.columns.to_flat_index(
    ).str.join('_')

    # drop columns
    bureau_merged_agg['SK_ID_BUREAU_count'] = bureau_merged_agg[
        'SK_ID_BUREAU_count_count']

    bureau_merged_agg = bureau_merged_agg.drop(columns=[
        'SK_ID_BUREAU_count_count', 'SK_ID_BUREAU_count_mean',
        'SK_ID_BUREAU_count_max', 'SK_ID_BUREAU_count_min',
        'SK_ID_BUREAU_count_std', 'SK_ID_BUREAU_count_nunique'
    ])

    # return final df of merged bureau and bureau_balance with new and encoded features
    return bureau_merged_agg.replace(to_replace=np.inf,value=np.nan).replace(to_replace=-np.inf,value=np.nan)
