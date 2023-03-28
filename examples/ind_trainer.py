from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import vectorbt as vbt
import pandas as pd

import xgboost as xgb
import numpy as np
import sys
import os
import glob
import pickle


file_path = "/home/steven/av_data"
#
# label: closes, tops, bottoms, indicators
#
def save_model(model, model_name):
    modeldir = os.path.join(file_path, "models")
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    file_name = os.path.join(modeldir, f'{model_name}.pkl')
    
    pickle.dump(model, open(file_name, "wb"))

def load_model(model_name):
    modeldir = os.path.join(file_path, "models")
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    file_name = os.path.join(modeldir, f'{model_name}.pkl')
    
    model_loaded = pickle.load(open(file_name, "rb"))
    return model_loaded
    
def save_data(ticker, df, tf='D', label='indicators'):
    intervaldir = os.path.join(file_path, 'training/interval-'+ tf)
    if not os.path.exists(intervaldir):
        os.makedirs(intervaldir)
    else:
        print(f"{intervaldir} exists.")
    
#     stockdir = os.path.join(intervaldir, ticker)
#     if not os.path.exists(stockdir):
#         os.makedirs(stockdir)
        
    to_save =  os.path.join(intervaldir,f"{ticker}_{label}.csv")
    print(f"[+] Saving: {to_save}")
#     csv = df.to_csv(to_save, header=None)
    csv = df.to_csv(to_save)
    return csv

def load_data(ticker, tf='D', label='indicators'):
    intervaldir =os.path.join(file_path, 'training/interval-'+tf)
    current_file =  os.path.join(intervaldir,f"{ticker}_{label}.csv")
#     stockdir = os.path.join(intervaldir, ticker)
#     current_file = f"{stockdir}/{label}.csv"
    
    if not os.path.exists(current_file):
        return pd.DataFrame()
#     df = pd.read_csv(current_file, index_col = 0, header=None)
    df = pd.read_csv(current_file, index_col = 0)
    return df

def load_folder(path=file_path, tf='D', label = 'indicators'):
    # Get CSV files list from a folder
    intervaldir = os.path.join(path, 'training/interval-'+ tf)
    csv_files = glob.glob(intervaldir + "/*.csv")

    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    df_list = (pd.read_csv(file) for file in csv_files)

    # Concatenate all DataFrames
    big_df   = pd.concat(df_list, ignore_index=True)
    return big_df

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def save_training_ma(ticker, df, col_name, tf, label, y):
    i = 1
#     print(df.shape)
    for x in np.arange(5,205, step=5, dtype=int):
#         print(vbt.MA.run(df, x).ma.fillna(0))
        df.insert(i, f'ma-{x}',vbt.MA.run(df[col_name], x).ma.fillna(0))
        i = i + 1
    df.insert(i, 'y',y)
    save_data(ticker, df, tf=tf, label=label)

def save_training_ma_delta(ticker, df, col_name, tf, label, y):
    i = 1
    gap = 1
    df.insert(i, f'prices-delta',df[col_name].fillna(0).diff())
    df.insert(i, f'log_return',np.log(df[col_name].fillna(0)).diff())
    i=i+1        
    for x in np.arange(1, 10, step=1, dtype=int):
        for z in np.arange(x+1, x+5, step = 1, dtype=int):
            if (x==1):
                df.insert(i, f'ma-prices-{z}',df[col_name].fillna(0) - vbt.MA.run(df[col_name], z).ma.fillna(0))
            else:
                df.insert(i, f'ma-{x}-{z}',vbt.MA.run(df[col_name], x).ma.fillna(0) - vbt.MA.run(df[col_name], z).ma.fillna(0))
            i = i + 1
    df.insert(i, 'y',y)
    save_data(ticker, df, tf=tf, label=label)

def save_training_ma_log(ticker, df, col_name, tf, label, y):
    i = 1
    gap = 1
    df.insert(i, f'prices-delta',np.log(df[col_name].fillna(0)).diff())
    i=i+1        
    for x in np.arange(1, 10, step=1, dtype=int):
        for z in np.arange(x+1, x+5, step = 1, dtype=int):
            if (x==1):
                df.insert(i, f'ma-prices-{z}',df[col_name].fillna(0) - vbt.MA.run(df[col_name], z).ma.fillna(0))
            else:
                df.insert(i, f'ma-{x}-{z}',vbt.MA.run(df[col_name], x).ma.fillna(0) - vbt.MA.run(df[col_name], z).ma.fillna(0))
            i = i + 1
    df.insert(i, 'y',y)
    save_data(ticker, df, tf=tf, label=label)
    
    
    
def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
    
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def train_ind(df, model, splits=5, mtype='classifcation'):
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]  
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)
    CV_SPLIT = 'time'  # 'time': time-series KFold 'group': GroupKFold by stock-id

    scores = []
#     print(X)
#     params = {
#         'max_depth': 6,
#         'objective': 'multi:softmax',  # error evaluation for multiclass training
#         'num_class': 3,
#         # Set number of GPUs if available   
#         'n_gpus': 0
#     }

    for train_index, test_index in kfold.split(X):          
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if (mtype == 'linear'):
            scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        else:
            scores.append(accuracy_score(y_test, y_pred)*100)

    return model, scores
    