from sklearn.metrics import roc_auc_score,roc_curve,scorer
import s3fs
import boto3
from io import StringIO
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def auc_group(test):
    try:
        return roc_auc_score(test.y, test.y_hat)
    except ValueError:
        return 0

def kfold_auc(test):

    df_avg_auc_each_fold = test.groupby(["Dataset", "Algo", "Fold"]).apply(auc_group)

    df_avg_auc = pd.DataFrame(df_avg_auc_each_fold).groupby(["Dataset", "Algo"])[0].mean()
    df_std_auc = pd.DataFrame(df_avg_auc_each_fold).groupby(["Dataset", "Algo"])[0].std()

    df_auc = pd.concat([df_avg_auc, df_std_auc], axis=1)
    df_auc.columns = ["mean_auc", "std_auc"]
    df_auc.reset_index(inplace=True)

    return df_auc[["Dataset", "Algo", "mean_auc"]].pivot(index="Algo", columns="Dataset", values="mean_auc")

def do_report(index_nb, path):
    """
    return a DF where each row is a classifier and each column a dataset
    """
    reporting = list()

    #for object_summary in my_bucket.objects.filter(Prefix="churn_package/churn/scoring_v2"):
    #    reporting.append(object_summary.key)
        
    for object_summary in my_bucket.objects.filter(Prefix=path[1:]):
        reporting.append(object_summary.key)        
    print(reporting)    
        
    # why I have to set [1:]
    # reporting = reporting[1:]
    
    sep = ";"
    
    dataset_names = reporting[index_nb].split("/")[-1]
    print(dataset_names)
    print(path)

    test = utils._read_dataframe_from_csv_on_s3(path, dataset_names, sep)

    df_result_kfold = kfold_auc(test)

    return df_result_kfold

def summary_report(verbose = True):
    """
    return a summary on each dataset where as column we have:
    ds_name	Nb. Instances	Nb. var dumi	perc_churner	perc_non_churner	ratio_feature/obs	ratio_churn/noChurn
    and as rows each dataset
    """
    # ------------ Report ------------ #
    l_feature_ratio = list()
    l_imb_ratio = list()
    l_dataset_name = list()
    l_nb_instance = list()
    l_nb_var = list()
    l_perc_churner = list()
    l_perc_non_churner = list()
    
    sep = ";"

    for ds in dataset_name:
        df_name_to_save = ds.split("/")[-1]
        
        if verbose == True:
            print(f"--- {df_name_to_save} ---")

        temp = utils._read_dataframe_from_csv_on_s3("", ds, sep)

        l_dataset_name.append(df_name_to_save.split(".")[0])

        # ratio feature / number of samples
        l_feature_ratio.append(round((temp.shape[1] / temp.shape[0]) * 100,2))

        l_nb_instance.append(temp.shape[0])
        l_nb_var.append(temp.shape[1])

        # imbalanced ratio
        if df_name_to_save == "KKBox.csv":
            l_imb_ratio.append(round((temp.is_churn.value_counts()[1] / temp.is_churn.value_counts()[0])*100,2))
            l_perc_churner.append(round(temp.is_churn.value_counts()[1] / (temp.is_churn.value_counts()[1] + temp.is_churn.value_counts()[0])*100,2))
            l_perc_non_churner.append(round(temp.is_churn.value_counts()[0] / (temp.is_churn.value_counts()[1] + temp.is_churn.value_counts()[0])*100,2))
        else:
            l_imb_ratio.append(round((temp.churn.value_counts()[1] / temp.churn.value_counts()[0])*100,2))
            l_perc_churner.append(round(temp.churn.value_counts()[1] / (temp.churn.value_counts()[1] + temp.churn.value_counts()[0])*100,2))
            l_perc_non_churner.append(round(temp.churn.value_counts()[0] / (temp.churn.value_counts()[1] + temp.churn.value_counts()[0])*100,2))


    df_summary = pd.DataFrame({"ds_name" : l_dataset_name, "Nb. Instances": l_nb_instance, "Nb. var dumi":  l_nb_var, "perc_churner": l_perc_churner,  "perc_non_churner":l_perc_non_churner ,"ratio_feature/obs" : l_feature_ratio, "ratio_churn/noChurn" : l_imb_ratio})
    
    # transpose the dataframe
    test = df_summary.T

    # take the first row and set it as column names
    test.columns = test.iloc[0]

    # remove the fist rows which is basically the column name
    df_summary = test.ix[1:]

    df_summary
    
    return df_summary 


# ------ homemade .py ------ #
s3 = boto3.resource('s3')
s3.meta.client.download_file(os.environ["S3_BUCKET"], "churn_package/churn/utils.py", "utils.py")

import utils

# Step 1 : get list of files in reporting directory
dataset_name = list()

_bucket = 'brigad-internal-eu-west-1-service-jupyter-production'

s3 = boto3.resource('s3')
my_bucket = s3.Bucket(_bucket)

for object_summary in my_bucket.objects.filter(Prefix="churn_package/churn/post_preprocessing_files"):
    dataset_name.append(object_summary.key)
    
# remove the first element of the list    
dataset_name = dataset_name[1:]

# to display full width for top3variables
pd.options.display.max_colwidth = 100

# ------ homemade .py ------ #
s3 = boto3.resource('s3')
s3.meta.client.download_file(os.environ["S3_BUCKET"], "churn_package/churn/io/write_reporting.py", "write_reporting.py")

import write_reporting

# --- test --- #
# intent : il faut que le knn donne des bon résultat
# classif normal retourne des bons algos

cross_val_type = "skfold"

# 1. launch code without KNN and with smote
write_reporting.write_report(cross_val_type, dataset_name, test_mode = False, smote = True, knn_bool = False)


# 2. launch code with KNN and with smote
write_reporting.write_report(cross_val_type, dataset_name, test_mode = False, smote = True, knn_bool = True)


# 3. launch code without KNN 
write_reporting.write_report(cross_val_type, dataset_name, test_mode = False, smote = False, knn_bool = False)


# 4. launch code with KNN 
write_reporting.write_report(cross_val_type, dataset_name, test_mode = False, smote = False, knn_bool = True)
