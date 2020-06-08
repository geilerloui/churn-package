import s3fs
import boto3
from io import StringIO
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score

def _give_columns_with_nan(df):
    """
    Read a dataframe from a CSV on S3
    Args:
        df: dataframe
    Returns:
        a list of columns with how many nan by columns
    Example:       
    """
    return df[[i for i in df.columns if df[i].isnull().any()]].isnull().sum()


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
    
    
def _write_report_to_csv(df, num_cols, bin_cols, multi_cols, target_col, dataset):
    """
    Write a dataframe to a CSV on S3 for the reporting table
    Args:
        df(pd.DataFrame): dataset you are working on
        num_cols(list): list of all numerical features
        bin_cols(list): list of all binary features
        multi_cols(list): list of all multicategorical features
        target_col(string): name of the target variable
        dataset(string): name of the dataset
    Returns:
        None
    Example:
    utils._write_report_to_csv(telcom.copy(), num_cols, bin_cols, multi_cols, target_col, dataset)
    """
    # ----------- Store values ----------- #

    n, d = df.shape
    list_numeric_cols = num_cols[:]
    list_binary_cols = bin_cols[:]
    list_multicat_cols = multi_cols[:]

    # percentage of churner // count
    perc_no_churn, perc_yes_churn = df[target_col[0]].value_counts(normalize=True)
    nb_no_churn, nb_yes_churn = df[target_col[0]].value_counts(normalize=False)

    # percentage of nan
    perc_nan = np.mean(df.isna().sum()/n)

    # number of feature with nan values
    nb_feature_with_nan = sum(df.isnull().sum() >0)
    # ----------- Writing to report file ----------- #
 
    row = [dataset, n, d,len(list_numeric_cols), len(list_binary_cols) ,len(list_multicat_cols)\
       ,nb_no_churn, nb_yes_churn, round(perc_no_churn,2)*100, round(perc_yes_churn,2)*100,\
       nb_feature_with_nan ,round(np.average(perc_nan),2)*100]
    
    col_summary = ["Dataset", "No. of obs", "No. of variables","No. of var. numeric",\
               "No. of binary var.","No. of multiclass var." ,"No. of non-churner",\
               "No. of churners", "% of non-churners", "% of churner",\
               "No. of features with nan", "avg. perc. nan"]
    df_report = pd.DataFrame(columns = col_summary)

    df_report.loc[row[1]] = row

    _write_dataframe_to_csv_on_s3(df_report, f'Churn/reporting/dashboard-overview/{row[0]}.csv')
    
    
def preprocess(df, num_cols, bin_cols, multi_cols, verbose=True):
    """
    Preprocess the DF for ML
    Args:
        df(pd.DataFrame): dataset you are working on
        num_cols(list): list of all numerical features
        bin_cols(list): list of all binary features
        multi_cols(list): list of all multicategorical features
        verbose(string): if you want to print a head on the final DF
    Returns:
        None
    Example:
    telcom = preprocess(telcom.copy(), num_cols, bin_cols, verbose=True)    
    """    
    #Label encoding Binary columns, which means if two values M, F -> 0, 1
    # here is to convert object type to int
    le = LabelEncoder()
    for i in bin_cols :
        df[i] = le.fit_transform(df[i])
    
    #Duplicating columns for multi value columns
    df = pd.get_dummies(data = df,columns = multi_cols )

    # Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(df[num_cols])
    scaled = pd.DataFrame(scaled,columns=num_cols)

    # dropping original values merging scaled values for numerical columns
    df = df.drop(columns = num_cols,axis = 1)
    df = df.merge(scaled,left_index=True,right_index=True,how = "left")
    
    df.columns = map(str.lower, df.columns)

    if verbose == True:
        print(df.head())
    
    return df

def telecom_churn_prediction_alg(algorithm,training_x,testing_x,
                                 training_y,testing_y,threshold_plot = True) :
    """
    it is used for SVC rbf / Gaussian NB / KNN
    Args:
        algorithm(string): algorithm name like "SVC-linear"
        training_x(pd.DataFrame): training set X
        testing_x(pd.DataFrame): testing set X
        training_y(pd.DataFrame): training set target
        testing_y(pd.DataFrame): testing set target
        threshold_plot: not used // to remove
    Returns:
        classification_report(dict): classif results - accuracy, recall/precision 0/1 and weighted 
        accuracy_score(np.float64): accuracy score
        model_roc_auc(np.float64): roc results
        ["empty"]: because there are no feature importance with those algorithms
    Example:
    telcom = preprocess(telcom.copy(), num_cols, bin_cols, verbose=True)  
    {'0': {'precision': 0.6153846153846154,
      'recall': 0.7540983606557377,
      'f1-score': 0.6777163904235727,
      'support': 244},
     '1': {'precision': 0.7014925373134329,
      'recall': 0.55078125,
      'f1-score': 0.6170678336980306,
      'support': 256},
     'accuracy': 0.65,
     'macro avg': {'precision': 0.6584385763490241,
      'recall': 0.6524398053278688,
      'f1-score': 0.6473921120608017,
      'support': 500},
     'weighted avg': {'precision': 0.65947187141217,
      'recall': 0.65,
      'f1-score': 0.6466643293800952,
      'support': 500}}
    """       
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy Score   : ",accuracy_score(testing_y,predictions))
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print ("Area under curve : ",model_roc_auc)
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
     
    return classification_report(testing_y,predictions, output_dict=True), \
accuracy_score(testing_y,predictions), model_roc_auc, ["empty"]

def telecom_churn_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf,threshold_plot, stacked=False) :
    """
    it is used for Logistic Regression / SVC linear / xgboost / RF
    Args:
        algorithm(string): algorithm name like "SVC-linear"
        training_x(pd.DataFrame): training set X
        testing_x(pd.DataFrame): testing set X
        training_y(pd.DataFrame): training set target
        testing_y(pd.DataFrame): testing set target
        cols(list): list of columns without the target
        cf(string): either "coefficients" either "features" 
        threshold_plot: not used
    Returns:
        classification_report(dict): classif results - accuracy, recall/precision 0/1 and weighted 
        accuracy_score(np.float64): accuracy score
        model_roc_auc(np.float64): roc results
        coef_sumry(list): list of top features
    Example:
    of classification_report
    {'0': {'precision': 0.6153846153846154,
      'recall': 0.7540983606557377,
      'f1-score': 0.6777163904235727,
      'support': 244},
     '1': {'precision': 0.7014925373134329,
      'recall': 0.55078125,
      'f1-score': 0.6170678336980306,
      'support': 256},
     'accuracy': 0.65,
     'macro avg': {'precision': 0.6584385763490241,
      'recall': 0.6524398053278688,
      'f1-score': 0.6473921120608017,
      'support': 500},
     'weighted avg': {'precision': 0.65947187141217,
      'recall': 0.65,
      'f1-score': 0.6466643293800952,
      'support': 500}}
    """       
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    if stacked == True:
        print (algorithm)
        print ("\n Classification report : \n",classification_report(testing_y,predictions))
        print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
        #confusion matrix
        conf_matrix = confusion_matrix(testing_y,predictions)
        #roc_auc_score
        model_roc_auc = 0 

        coef_sumry = "empty"
        
    else:
        probabilities = algorithm.predict_proba(testing_x)
        #coeffs
        if   cf == "coefficients" :
            coefficients  = pd.DataFrame(algorithm.coef_.ravel())
        elif cf == "features" :
            coefficients  = pd.DataFrame(algorithm.feature_importances_)

        column_df     = pd.DataFrame(cols)
        coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                                  right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

        print(f"{coef_sumry[0:3]} \n")

        print (algorithm)
        print ("\n Classification report : \n",classification_report(testing_y,predictions))
        print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
        #confusion matrix
        conf_matrix = confusion_matrix(testing_y,predictions)
        #roc_auc_score
        model_roc_auc = roc_auc_score(testing_y,predictions) 
        print ("Area under curve : ",model_roc_auc,"\n")
        fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
       

    return classification_report(testing_y,predictions, output_dict=True), \
accuracy_score(testing_y,predictions), model_roc_auc, coef_sumry