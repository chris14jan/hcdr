from data.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data = Data()
df_dict = data.get_data(nrows=300_000)
df = df_dict['bureau']
df_bal = df_dict['bureau_balance']


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


def get_final_bureau_merged(df, df_bal):
    """This function calls all previous functions to get the final merged df"""

    bureau = get_bureau_final(df)
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
    return bureau_merged_agg
