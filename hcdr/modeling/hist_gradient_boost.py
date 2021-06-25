# packages
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, plot_roc_curve

# import functions
from hcdr.data.data import Data
from hcdr.modeling.preproc import preproc_pipeline
from hcdr.data.merged_data import merge_dfs

# import model
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


def get_train_test_data():
    # preprocessor
    preproc = preproc_pipeline(scaler_type=None)

    # merge train dataframes
    df_merged = merge_dfs(df_app="application_train", verbose=True)
    df_merged = df_merged.replace(to_replace=np.inf,
                                  value=np.nan).replace(to_replace=-np.inf,
                                                        value=np.nan)
    # define train variables
    X = df_merged.drop(columns=["SK_ID_CURR", "TARGET"])
    X_transformed = preproc.fit_transform(X)
    y = df_merged["TARGET"]

    # merge test dataframes
    df_test = merge_dfs(df_app="application_test", verbose=True)
    # define test variables
    df_test = df_test.replace(to_replace=np.inf,
                              value=np.nan).replace(to_replace=-np.inf,
                                                    value=np.nan)
    test = df_test.drop(columns=["SK_ID_CURR"])
    y_test = preproc.transform(test)

    return X_transformed, y, df_test, y_test


def hgb_model():

    # define variables
    # train data
    train_test = get_train_test_data()
    X_transformed = train_test[0]
    y = train_test[1]
    # test data
    df_test = train_test[2]
    y_test = train_test[3]

    # instantiate model
    hgb = HistGradientBoostingClassifier(max_depth=15,
                                         max_iter=20_000,
                                         scoring='roc_auc',
                                         verbose=1)
    model_hgb = hgb.fit(X_transformed, y)

    # save final model to pickle file
    filename = 'finalized_hgb_model.sav'
    pickle.dump(model_hgb, open(filename, 'wb'))

    # run prediction
    hgb_pred = model_hgb.predict_proba(y_test)[:, 1]

    # create submission file
    submit = df_test[['SK_ID_CURR']]
    submit['TARGET'] = hgb_pred

    # save submission file
    submit.to_csv('hgb_hcdr.csv', index=False)

    return submit.head()
