from hcdr.modeling.preproc import preproc_pipeline
from hcdr.data.merged_data import training_data, test_data
from hcdr.data.data import Data
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from pickle import dump, load
import csv
import os
from pathlib import Path

def undersample(X_train, y_train, undersampling_strategy=0.2):
    undersample = RandomUnderSampler(sampling_strategy=undersampling_strategy, replacement=True)
    X_train_undersampled, X_train_undersampled = undersample.fit_resample(X_train, y_train)
    return X_train_undersampled, X_train_undersampled

def preprocess_data(scaler_type="standard", load_saved=False):
    root_dir = Path(__file__).parents[1]
    
    print("LOADING TRAIN DATA")
    
    # df_merged_path = os.path.join(root_dir, f"data/merged_tables/df_merged_train.pkl")
    # df_merged = pd.read_pickle(df_merged_path)
    
    # print("SPLITTING MERGED DF INTO X and Y...")
    # X_train = df_merged.drop(columns=["SK_ID_CURR", "TARGET"])
    # y_train = df_merged["TARGET"]
    
    X_train, y_train = training_data(load_saved=load_saved)
    
    path_X_train = os.path.join(root_dir, f"modeling/modeling_data_dl/X_train_xformed.pkl")
    path_y_train = os.path.join(root_dir, f"modeling/modeling_data_dl/y_train_xformed.pkl")
    path_X_test = os.path.join(root_dir, f"modeling/modeling_data_dl/X_test_xformed.pkl")
    
    print("save y_train")
    y_train.to_pickle(path_y_train)
    del y_train
    
    print("Creating the preprocessor...")
    preprocessor = preproc_pipeline(scaler_type=scaler_type)
    
    print("Fitting the preprocessor...")
    preprocessor.fit(X_train)
    path_preproc = os.path.join(root_dir, f"modeling/modeling_data_dl/preproc_dl")
    
    print("Pickling the preprocessor...")
    print(f"Saved to: {path_preproc} \n Saving...")
    dump(preprocessor, open(path_preproc, "wb" ))
    
    print("Transforming X_train thru fitted pre_processor AND deleting original X_train var...")
    X_train_xformed = preprocessor.transform(X_train)
    del X_train
    
    print("save X_train")
    pd.DataFrame(X_train_xformed).to_pickle(path_X_train)
    del X_train_xformed
    
    print("LOADING TEST DATA")
    X_test = test_data(load_saved=load_saved)
    print("Transforming X_test thru fitted pre_processor AND deleting original X_test var...")
    X_test_xformed = preprocessor.transform(X_test)
    del X_test
    
    pd.DataFrame(X_test_xformed).to_pickle(path_X_test)
    del X_test_xformed
    
    return None

def split_data():
    root_dir = Path(__file__).parents[0]
    path_X_train = os.path.join(root_dir, f"modeling_data_dl/X_train_xformed.pkl")
    
    print("Loading saved X_train and y_train")
    path_X_train = os.path.join(root_dir, f"modeling_data_dl/X_train_xformed.pkl")
    path_y_train = os.path.join(root_dir, f"modeling_data_dl/y_train_original.pkl")
    
    X_train = pd.read_pickle(path_X_train)
    y_train = pd.read_pickle(path_y_train)
    
    # First Split:
    X_train, X_test_model, y_train, y_test_model = train_test_split(X_train, y_train, test_size=0.3)
    
    path_X_test_model = os.path.join(root_dir, f"modeling_data_dl/X_test_model.pkl")
    pd.DataFrame(X_test_model).to_pickle(path_X_test_model)
    del X_test_model
    
    path_y_test_model = os.path.join(root_dir, f"modeling_data_dl/y_test_model.pkl")
    pd.DataFrame(y_test_model).to_pickle(path_y_test_model)
    del y_test_model
    
    
    # Second Split:
    X_train_model, X_val_model, y_train_model, y_val_model = train_test_split(X_train, y_train, test_size=0.3)
    path_X_val_model = os.path.join(root_dir, f"modeling_data_dl/X_val_model.pkl")
    pd.DataFrame(X_val_model).to_pickle(path_X_val_model)
    del X_val_model
    
    path_y_val_model = os.path.join(root_dir, f"modeling_data_dl/y_val_model.pkl")
    pd.DataFrame(y_val_model).to_pickle(path_y_val_model)
    del y_val_model
    
    # Save X, y train before undersampling:
    path_X_train_model = os.path.join(root_dir, f"modeling_data_dl/X_train_model.pkl")
    pd.DataFrame(X_train_model).to_pickle(path_X_train_model)
    path_y_train_model = os.path.join(root_dir, f"modeling_data_dl/y_train_model.pkl")
    pd.DataFrame(y_train_model).to_pickle(path_y_train_model)
    
    # Undersampled X, y train:
    X_train_model_under, y_train_model_under = undersample_majority(X_train_model, y_train_model, undersampling_strategy=0.2)
    del X_train_model
    del y_train_model
    
    path_X_train_model_under = os.path.join(root_dir, f"modeling_data_dl/X_train_model_under.pkl")
    pd.DataFrame(X_train_model_under).to_pickle(path_X_train_model_under)
    del X_train_model_under
    
    path_y_train_model_under = os.path.join(root_dir, f"modeling_data_dl/y_train_model_under.pkl")
    pd.DataFrame(y_train_model_under).to_pickle(path_y_train_model_under)
    del y_train_model_under
    
    return None

def undersample_majority(X, y, undersampling_strategy=0.2):
    """undersampling_strategy : flt (# of minority / # of majority) ratio. 
        Try 0.2 as a starting point. 
        0.1 is close to the true ratio. 
        Greater than 0.3 greatly reduces the number of rows to model."""
        
    print("Undersampling the majority class...")
    X_train, y_train = undersample(X, y, undersampling_strategy=undersampling_strategy)
    return X_train, y_train

def initialize_model(input_dim, neurons_list=None):
    """"""
    if neurons_list == None:
        neurons_list=[16, 16, 16, 4]
    
    model = models.Sequential()
    
    for i, neurons in enumerate(neurons_list):
        if i==0:
            model.add(layers.Dense(neurons, activation='relu',
                                   input_dim=input_dim,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2=0.01), # regularizes the learning rate
                                   kernel_initializer="he_normal")) # how the initial weights are randomly generated
            
            print(f"added layer {i} with {neurons} neurons. input_dim={input_dim}")
            continue
        else:
            model.add(layers.Dense(neurons, activation='relu'))
            print(f"added layer {i} with {neurons} neurons.")
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def compile_model(model):
    ### Model optimization : Optimizer, loss and metric 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(), # "binary_crossentropy"
                  metrics=[tf.keras.metrics.AUC(), "accuracy"]) # "accuracy"
    return model
    

def train_model(neurons_list=None, pre_split=False, validation_split=0.3, undersampling=False, dict_model_info={}):
    root_dir = Path(__file__).parents[0]
    
    
    print("Loading saved X_train...")
    path_X_train = os.path.join(root_dir, f"modeling_data_dl/X_train_xformed.pkl")
    print("Loading saved y_train...")
    path_y_train = os.path.join(root_dir, f"modeling_data_dl/y_train_original.pkl")
    
    X_train = pd.read_pickle(path_X_train)
    y_train = pd.read_pickle(path_y_train)
    
    # print("Splitting Test Model Data...")
    # X_train, X_test_model, y_train, y_test_model = train_test_split(X_train, y_train, test_size=0.2)
    
    validation_data=None
    if pre_split:
        print(f"Splitting train data into train and validation data...  \n validation_size={validation_split}")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split)
        validation_data=(X_val, y_val)
        del X_val
        del y_val
        validation_split=0.0
        
    if undersampling and pre_split:
        print("Undersampling the majority category...")
        X_train, y_train= undersample_majority(X_train, y_train, undersampling_strategy=0.2)
    elif undersampling and not pre_split:
        print("WARNING: UNDERSAMPLING OF THE MAJORITY CATEGORY OF THE VALIDATION DATA SET IS NOT RECOMMENDED.")
    
    print("Initializing model...")
    model = initialize_model(input_dim=X_train.shape[1], neurons_list=neurons_list)
    print("Compiling model...")
    model = compile_model(model)
    
    patience = 2
    es = EarlyStopping(patience=patience, restore_best_weights=True)
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience-1,)
    
    print("Fitting model...")
    history = model.fit(X_train, y_train,
                    validation_data=validation_data,
                    validation_split=validation_split,
                    epochs=1_000, 
                    batch_size=128, 
                    verbose=1,
                    callbacks=[es]
                    )
    
    AUC_score_val = history.history["val_auc"][-1]
    # AUC_score_test = model.evaluate(X_test_model, y_test_model)[1]
        
    model_id = len(dict_model_info)+1
    path_model = os.path.join(root_dir, f"fitted_models_dl/tf_model_"+str(model_id)+".h5")
    model.save(path_model)
    
    print(model_id, AUC_score_val)
    
    dict_model_info = {model_id:{"neurons_list" : neurons_list, 
                       #"AUC_score_val": AUC_score_val,
                        #"AUC_score_test": AUC_score_test,
                        "pre_split" : pre_split,
                        "validation_split" : 0.3, #validation_split
                        "undersampling" : undersampling,
                        "patience" : patience}}
    
    # with open('models.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    #     w = csv.DictWriter(f, dict_model_info[model_id].keys())
    #     w.writeheader()
    #     w.writerow(dict_model_info[model_id])
    
    
    return dict_model_info

def prediction(X, model_id=1):
    root_dir = Path(__file__).parents[0]
    path_preproc = os.path.join(root_dir, f"modeling_data_dl/preproc_dl")
    preprocessor = load(open(path_preproc, "rb" ))
    
    X.drop(columns=["SK_ID_CURR"], inplace=True)
    X_xformed = preprocessor.transform(X)
    
    path_model = os.path.join(root_dir, f"fitted_models_dl/tf_model_"+str(model_id)+".h5")
    model = tf.keras.models.load_model(path_model, compile=True)
    model = compile_model(model)
    
    y_pred = model.predict(X_xformed)
    
    return y_pred


def create_submision(model_id=2, save_submission=False):
    root_dir = Path(__file__).parents[0]
    
    print("Loading saved model")
    print(root_dir)
    
    path_model = os.path.join(root_dir, f"fitted_models_dl/tf_model_"+str(model_id)+".h5")
    model = tf.keras.models.load_model(path_model, compile=True)
    model = compile_model(model)
    
    path_X_test_model = os.path.join(root_dir, f"modeling_data_dl/X_test_model.pkl")
    path_y_test_model = os.path.join(root_dir, f"modeling_data_dl/y_test_model.pkl")
    X_test_model = pd.read_pickle(path_X_test_model)
    y_test_model = pd.read_pickle(path_y_test_model)
    
    AUC_score_test = model.evaluate(X_test_model, y_test_model)[1]
    print(f"AUC_score_test = {AUC_score_test}")
    
    path_X_test = os.path.join(root_dir, f"modeling_data_dl/X_test_xformed.pkl")
    X_test = pd.read_pickle(path_X_test)
    y_test = model.predict(X_test)
    
    if save_submission:
        submission = Data().get_data(tables=["application_test"])["application_test"][["SK_ID_CURR"]]
        submission['TARGET'] = y_test
        path_submission = os.path.join(root_dir, f"submissions/submission_dl.csv")
        submission.to_csv(path_submission, index=False)
    
    return None

if __name__ == "__main__": 
    # train_model()
    # predict_test_data(model_id=1)
    preprocess_data(scaler_type=None)