from hcdr.data.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_credit_card_balance_df(agg="mean"): 
    
    """ A function that retrieves, preprocesses, feature engineers and groups the credit_card_balance_df by SK_ID_CURR using the mean """
    
    credit_card_balance_df = Data().get_data(tables=["credit_card_balance"])["credit_card_balance"]
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
    
    installments_payments_df = Data().get_data(tables=["installments_payments"])["installments_payments"]
    
    columns_to_drop = ["NUM_INSTALMENT_VERSION"]
    installments_payments_df = installments_payments_df.drop(columns=columns_to_drop)
    
    grouped_df = installments_payments_df.groupby("SK_ID_CURR").agg(agg)
    
    return grouped_df

def preprocess_POS_CASH_balance_df(agg="mean"): 
    
    """ A function that retrieves and groups the POS_CASH_balance_df by SK_ID_CURR using the mean"""
    
    POS_CASH_balance_df = Data().get_data(tables=["POS_CASH_balance"])["POS_CASH_balance"]
    
    columns_to_drop = []
    POS_CASH_balance_df = POS_CASH_balance_df.drop(columns= columns_to_drop)
    
    grouped_df = POS_CASH_balance_df.groupby("SK_ID_CURR").agg("mean")
    
    return grouped_df
