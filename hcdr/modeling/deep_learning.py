from hcdr.modeling import preproc
from hcdr.data.merged_data import merge_dfs

df_merged = merge_dfs(df_app="application_train", verbose=True)

X = df_merged.drop(columns=["SK_ID_CURR", "TARGET"]).iloc[:1000]
y = df_merged["TARGET"].iloc[:1000]

preproc = preproc(scaler_type=None)
X_transformed = preproc.fit_transform(X)

