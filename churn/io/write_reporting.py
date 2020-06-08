from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

import s3fs
import boto3
from io import StringIO
import os
import re
from itertools import chain

import time

from warnings import filterwarnings
filterwarnings('ignore')


def _read_dataframe_from_csv_on_s3(path, dataset_name, sep):
    """
    Read a dataframe from a CSV on S3
    Args:
        path(str):
        dataset_name(str):
        sep(str):
    Returns:
        df(pd.DataFrame): the file is returned as a DF
    Example:
        path = "/Churn/datasets"
        dataset_name = "CrowdAnalytix/telecom churn(original_data).csv"        
    """

    fs = s3fs.S3FileSystem(anon=False)

    with fs.open(os.environ['S3_BUCKET'] + f'{path}/{dataset_name}') as f:
             df = pd.read_table(f, sep=sep)
    return df

def _write_dataframe_to_csv_on_s3(dataframe, filename):
    """
    Write a dataframe to a CSV on S3 
    Args:
        dataframe(pd.DataFrame): dataframe to write
        filename(str): where to write the dataframe as a csv
    Returns:
        None
    Example:
        _write_dataframe_to_csv_on_s3(df, f'Churn/reporting/{dataset}.csv')       
    """
    
    bucket = 'brigad-internal-eu-west-1-service-jupyter-production'

    print("Writing {} records to {}".format(len(dataframe), filename))
    # Create buffer
    csv_buffer = StringIO()
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep=";", index=False)
    # Create S3 object
    s3_resource = boto3.resource("s3")
    # Write buffer to S3 object
    s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())
    
    
def algorithm(knn_bool):
    # --------- list of algorithms --------- #

    logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=-1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)

    logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=-1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)

    svc_lin  = SVC(C=1.0, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
                   max_iter=1000, probability=True, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)

    svc_rbf  = SVC(C=1.0, kernel='rbf', 
                   degree= 3, gamma=1.0, 
                   coef0=0.0, shrinking=True,
                   probability=True,tol=0.001,
                   class_weight=None,
                   verbose=False,max_iter= 1000,
                   random_state=None)

    xgc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                        max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,
                        n_jobs=-1, objective='binary:logistic', random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                        silent=True, subsample=1)

    gnb = GaussianNB(priors=None)

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
               weights='uniform')

    rf = RandomForestClassifier()

    dtc = DecisionTreeClassifier()

    stacked = VotingClassifier(estimators=[('logit_no_smote',logit),('logit',logit_smote), \
                                         ('SVM_linear', svc_lin), ("SVM_rbf", svc_rbf), ("Xgboost", xgc), \
                                         ('GNB', gnb), ("KNN", knn), ("RandomForest", rf),('DTC', dtc) 
                                        ])


    # prepare models for K-fold
    models = list()

    if knn_bool == False:
        models.append(('LR', logit_smote))
        models.append(('SVM', svc_lin))
        models.append(('SVM-rbf', svc_rbf))
        models.append(('Gnb', gnb))
        models.append(('RF', rf))
        models.append(('DT', dtc))
        models.append(('XGBoost', xgc))
    elif knn_bool == True:
        models.append(('KNN', knn))
        
    return models

def write_report(cross_val_type, dataset_name ,test_mode = False, smote = False, cross_val_stratify = False, knn_bool = False):
    models = algorithm(knn_bool)
    
    l_df = list()
    
    if test_mode == True:
        dataset_name = ['churn_package/churn/post_preprocessing_files/Bank.csv','churn_package/churn/post_preprocessing_files/TelC.csv']
    
    
    for df in dataset_name:
        sep = ";"
        path = "/churn_package/churn/post_preprocessing_files"
        dataset_names = df.split("/")[-1]
        dataset_names_for_report = dataset_names.split(".")[0]
        print(f"--- {dataset_names_for_report} ---")

        telcom = _read_dataframe_from_csv_on_s3(path, dataset_names, sep)

        if "churn status" in telcom.columns:    
            telcom.rename(columns={"churn status":"churn"}, inplace=True)
        elif "is_churn" in telcom.columns:
            telcom.rename(columns={"is_churn":"churn"}, inplace=True)
        
        # -- report dataframe
        if knn_bool == False:
            algo_index = ['logit', 'SVM-lin', 'SVM-rbf', 'XGB', 'GNB', 'RF', 'DTC', 'n', 'perc_churner']
        else:
            algo_index = [ 'KNN', 'n', 'perc_churner']
            
        report_df = pd.DataFrame(index = algo_index)
        report_df = pd.DataFrame()
        
        if cross_val_type == "skfold":
            target_col = ["churn"]

            skf = StratifiedKFold(n_splits=5, random_state = 0)
            
            X = telcom.drop("churn", axis=1)
            Y = telcom["churn"]

            results = []
            names = []

            to_store2 = list()

            seed = 0
            scoring = "roc_auc"

            cv_results = np.array([])
            avg_results = list()
            l_0 = list()
            l_1 = list()
            l_2 = list()
            l_3 = list()
            l_4 = list()
            l_5 = list()
            l_6 = list()
            
            fold_nb = 0

            for name, model in models:
                fold_nb = 1
                for train_index, test_index in skf.split(X, Y):
                    # begin
                    start = time.time()
                    
                    if smote == False:
                        # split the data
                        X_train, X_test = X.loc[train_index,:].values, X.loc[test_index,:].values
                        y_train, y_test = np.ravel(Y[train_index]), np.ravel(Y[test_index])  
                    elif smote == True:
                        # split the data
                        X_train, X_test = X.loc[train_index,:].values, X.loc[test_index,:].values
                        y_train, y_test = np.ravel(Y[train_index]), np.ravel(Y[test_index]) 
                        
                        sm = SMOTE(random_state=152)
                        X_train, y_train = sm.fit_sample(X_train, y_train)
                    
                    
                    model = model  # Choose a model here
                    model.fit(X_train, y_train )  
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # end
                    end = time.time()
                    running_time = round((end - start), 5)

                    # store fold results
                    result = roc_auc_score(y_test, y_pred)
                    cv_results = np.append(cv_results, result)
                    
                    l_0.extend(list(chain(*([n]*len(y_test) for n in [dataset_names_for_report]))))
                    l_1.extend(list(chain(*([n]*len(y_test) for n in [name]))))
                    l_2.extend(list(chain(*([n]*len(y_test) for n in [fold_nb]))))
                    # we add model running time
                    l_6.extend(list(chain(*([n]*len(y_test) for n in [running_time]))))
                    l_3.extend(y_pred_proba[:,1])
                    l_4.extend(y_pred)
                    l_5.extend(y_test)
                    print(f"fold number {fold_nb}")
                    fold_nb = fold_nb + 1
                
                results.append(cv_results)
                names.append(name)
                msg = "%s: %.4f (%.4f)" % (name, round(cv_results.mean(),4), round(cv_results.std(),4))
                print(msg)

            l_df.append(pd.DataFrame(list(zip(l_0, l_1, l_2, l_3, l_4, l_5, l_6)), columns=["Dataset", "Algo", "Fold", "p(y=1)", "y_hat", "y", "run_time"]))
         
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        elif cross_val_type == "crossval":
            target_col = ["churn"]
            
            X = telcom.drop("churn", axis=1)
            Y = telcom["churn"]

            results = []
            names = []

            to_store2 = list()

            seed = 0
            scoring = "roc_auc"

            cv_results = np.array([])
            avg_results = list()
            l_0 = list()
            l_1 = list()
            l_3 = list()
            l_4 = list()
            l_5 = list()
            
            fold_nb = 0

            for name, model in models:
                if smote == False:
                    if cross_val_stratify == False:
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
                    elif cross_val_stratify == True:
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)
                elif smote == True:
                    if cross_val_stratify == False:
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
                        sm = SMOTE(random_state=152)
                        X_train, y_train = sm.fit_sample(X_train, y_train)  
                    elif cross_val_stratify == True:
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)
                        sm = SMOTE(random_state=152)
                        X_train, y_train = sm.fit_sample(X_train, y_train) 

                model.fit(X_train, y_train )  
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)    
                # store fold results
                #result = roc_auc_score(y_test, y_pred)
                #cv_results = np.append(cv_results, result)

                l_0.extend(list(chain(*([n]*len(y_test) for n in [dataset_names_for_report]))))
                l_1.extend(list(chain(*([n]*len(y_test) for n in [name]))))
                l_3.extend(y_pred_proba[:,1])
                l_4.extend(y_pred)
                l_5.extend(y_test)  

            l_df.append(pd.DataFrame(list(zip(l_0, l_1, l_3, l_4, l_5)), columns=["Dataset", "Algo", "p(y=1)", "y_hat", "y"]))  
                
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  
        elif cross_val_type == "kfold":
            target_col = ["churn"]

            kf = KFold(n_splits=5, random_state = 0)
            
            X = telcom.drop("churn", axis=1)
            Y = telcom["churn"]

            results = []
            names = []

            to_store2 = list()

            seed = 0
            scoring = "roc_auc"

            cv_results = np.array([])
            avg_results = list()
            l_0 = list()
            l_1 = list()
            l_2 = list()
            l_3 = list()
            l_4 = list()
            l_5 = list()
            
            fold_nb = 0

            for name, model in models:
                fold_nb = 1
                for train_index, test_index in kf.split(X):
                    if smote == False:
                        # split the data
                        X_train, X_test = X.loc[train_index,:].values, X.loc[test_index,:].values
                        y_train, y_test = np.ravel(Y[train_index]), np.ravel(Y[test_index])  
                    elif smote == True:
                        # split the data
                        X_train, X_test = X.loc[train_index,:].values, X.loc[test_index,:].values
                        y_train, y_test = np.ravel(Y[train_index]), np.ravel(Y[test_index]) 
                        
                        sm = SMOTE(random_state=152)
                        X_train, y_train = sm.fit_sample(X_train, y_train)
                    
            
                    model = model  # Choose a model here
                    model.fit(X_train, y_train )  
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)

                    # store fold results
                    #result = roc_auc_score(y_test, y_pred)
                    #cv_results = np.append(cv_results, result)
                    
                    l_0.extend(list(chain(*([n]*len(y_test) for n in [dataset_names_for_report]))))
                    l_1.extend(list(chain(*([n]*len(y_test) for n in [name]))))
                    l_2.extend(list(chain(*([n]*len(y_test) for n in [fold_nb]))))
                    l_3.extend(y_pred_proba[:,1])
                    l_4.extend(y_pred)
                    l_5.extend(y_test)
                    print(f"fold number {fold_nb}")                    
                    fold_nb = fold_nb + 1


                #results.append(cv_results)
                #names.append(name)
                #msg = "%s: %.4f (%.4f)" % (name, round(cv_results.mean(),4), round(cv_results.std(),4))
                #print(msg)

            l_df.append(pd.DataFrame(list(zip(l_0, l_1, l_2, l_3, l_4, l_5)), columns=["Dataset", "Algo", "Fold", "p(y=1)", "y_hat", "y"]))
         
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                
    if knn_bool == True:
        if cross_val_type == "skfold":
            if smote == False:
                path_out = f"churn_package/churn/scorings/scorings_knn_stratified.csv"
            elif smote == True:
                path_out = f"churn_package/churn/scorings/scorings_knn_stratified_smote.csv"
        elif cross_val_type == "crossval":
            if smote == False:
                if cross_val_stratify == False:
                    path_out = f"churn_package/churn/scorings/scorings_knn_cross_val.csv"
                elif cross_val_stratify == True:
                    path_out = f"churn_package/churn/scorings/scorings_knn_cross_val_stratified.csv"
            elif smote == True:
                if cross_val_stratify == False:
                    path_out = f"churn_package/churn/scorings/scorings_knn_cross_val_smote.csv"
                elif cross_val_stratify == True:
                    path_out = f"churn_package/churn/scorings/scorings_knn_cross_val_smote_stratified.csv" 
        elif cross_val_type == "kfold":
            if smote == False:
                path_out = f"churn_package/churn/scorings/scorings_knn_kfold.csv"
            elif smote == True:
                path_out = f"churn_package/churn/scorings/scorings_knn_kfold_smote.csv"        
    else:
        if cross_val_type == "skfold":
            if smote == False:
                path_out = f"churn_package/churn/scorings/scorings_stratified.csv"
            elif smote == True:
                path_out = f"churn_package/churn/scorings/scorings_stratified_smote.csv"
        elif cross_val_type == "crossval":
            if smote == False:
                if cross_val_stratify == False:
                    path_out = f"churn_package/churn/scorings/scorings_cross_val.csv"
                elif cross_val_stratify == True:
                    path_out = f"churn_package/churn/scorings/scorings_cross_val_stratified.csv"
            elif smote == True:
                if cross_val_stratify == False:
                    path_out = f"churn_package/churn/scorings/scorings_cross_val_smote.csv"
                elif cross_val_stratify == True:
                    path_out = f"churn_package/churn/scorings/scorings_cross_val_smote_stratified.csv" 
        elif cross_val_type == "kfold":
            if smote == False:
                path_out = f"churn_package/churn/scorings/scorings_kfold.csv"
            elif smote == True:
                path_out = f"churn_package/churn/scorings/scorings_kfold_smote.csv"
                
    df_out = pd.concat(l_df, axis=0)
    _write_dataframe_to_csv_on_s3(df_out, path_out)
    
    
    