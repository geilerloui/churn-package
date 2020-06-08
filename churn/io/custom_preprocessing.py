import s3fs
import boto3
from io import StringIO
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def create_preprocessing_files():

    # to display full width for top3variables
    pd.options.display.max_colwidth = 100

    # ------ homemade .py ------ #
    s3 = boto3.resource('s3')
    s3.meta.client.download_file(os.environ["S3_BUCKET"], "churn_package/churn/utils.py", "utils.py")

    import utils

    # Step 1 : get list of files in reporting directory
    dataset_name = list()

    _bucket = 'brigad-internal-eu-west-1-service-jupyter-production'

    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(_bucket)

    for object_summary in my_bucket.objects.filter(Prefix="churn_package/datasets"):
        dataset_name.append(object_summary.key.split("/")[-2])

    dataset_name.remove('datasets')
    dataset_name = list(set(dataset_name))

    for dataset in dataset_name:    
        if dataset == "KDD-cup-2015-MOOC":
            pass
        elif dataset == 'telecom-IBM-watson':
            pass
        elif dataset == 'churn-telco-europa':
            path = "/churn_package/datasets"

            dataset_name = "churn-telco-europa/train_churn_kg.csv"
            sep = ","

            # data preprocessing
            target_col = ["CHURN"]

            # data for writing file
            dataset = "TelE"

            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            # --------- Unique preprocessing for this data set --------- #

            # None

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            # --------- general preprocessing --------- #

            del telcom["CETEL_NUMBER"]
            del telcom["CNI_CUSTOMER"]


            num_cols = ['DAYS_LIFE',
             'DEVICE_TECNOLOGY',
             'MIN_PLAN',
             'PRICE_PLAN',
             'TOT_MIN_CALL_OUT',
             'AVG_MIN_CALL_OUT_3',
             'TOT_MIN_IN_ULT_MES',
             'AVG_MIN_IN_3',
             'ROA_LASTMONTH',
             'ROACETEL_LAST_MONTH',
             'DEVICE',
             'STATE_DATA',
             'CITY_DATA',
             'STATE_VOICE',
             'CITY_VOICE']

            multi_cols = ['TEC_ANT_DATA', 'TEC_ANT_VOICE']

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)

            target_col = ["churn"]

            # ----- post preprocessing ----- #
            telcom.fillna(telcom.mean(), inplace=True)

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)

        elif dataset == "Cell2Cell":
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "Cell2Cell/cell2cell-duke univeristy.csv"
            sep = ","

            # data preprocessing
            target_col = ["churn"]

            # data for writing file
            dataset = "C2C"
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)

            # Unknown in the dataset is a NaN
            telcom.replace('Unknown', np.nan, inplace=True)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            #Replacing blank spaces with null values in total charges column
            del telcom['Unnamed: 0']

            del telcom['X']   

            telcom.fillna(telcom.mean(), inplace=True)
            del telcom["traintest"]

            del telcom["churndep"]

            del telcom["customer"]


            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]        
            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)
            utils._write_dataframe_to_csv_on_s3(telcom, path_out)


        elif dataset == "KKBox":
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name_train = "KKBox/train_v2.csv"
            dataset_name_transaction = "KKBox/transactions_v2.csv"
            dataset_name_log = "KKBox/user_logs_v2.csv"
            dataset_name_mem = "KKBox/members_v3.csv"

            sep = ","

            # data preprocessing
            target_col = ["Churn"]

            # data for writing file
            dataset = "KKBox"

            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # train
            train = utils._read_dataframe_from_csv_on_s3(path, dataset_name_train, sep)
            #train.columns = ["msno", "is_churn"]

            # transaction
            transa = utils._read_dataframe_from_csv_on_s3(path, dataset_name_transaction, sep)

            # log
            log = utils._read_dataframe_from_csv_on_s3(path, dataset_name_log, sep)

            # member
            member = utils._read_dataframe_from_csv_on_s3(path, dataset_name_mem, sep)

            train = pd.merge(train, member, on="msno", how="left")
            del member
            train = pd.merge(train,transa,how='left',on='msno',left_index=True, right_index=True)
            del transa
            train = pd.merge(train,log,how='left',on='msno',left_index=True, right_index=True)
            del log
            train['registration_init_time'] = train['registration_init_time'].fillna(value='20151009')
            train["transaction_date"] = pd.to_datetime(train["transaction_date"])
            train["date"] = pd.to_datetime(train["date"])
            train["membership_expire_date"] = pd.to_datetime(train["membership_expire_date"])
            train["registration_init_time"] = pd.to_datetime(train["registration_init_time"])

            def date_feature(df):

                col = ['registration_init_time' ,'transaction_date','membership_expire_date','date']
                var = ['reg','trans','mem_exp','user_']
                #df['duration'] = (df[col[1]] - df[col[0]]).dt.days 

                for i ,j in zip(col,var):
                    df[j+'_day'] = df[i].dt.day.astype('uint8')
                    df[j+'_weekday'] = df[i].dt.weekday.astype('uint8')        
                    df[j+'_month'] = df[i].dt.month.astype('uint8') 
                    df[j+'_year'] =df[i].dt.year.astype('uint16') 

            date_feature(train)

            col = [ 'city', 'bd', 'gender', 'registered_via']
            def missing(df,columns):
                col = columns
                for i in col:
                    df[i].fillna(df[i].mode()[0],inplace=True)

            missing(train,col)

            le = LabelEncoder()
            train['gender'] = le.fit_transform(train['gender'])

            def OHE(df):
                #col = df.select_dtypes(include=['category']).columns
                col = ['city','gender','registered_via']
                #print('Categorical columns in dataset',col)

                c2,c3 = [],{}
                for c in col:
                    if df[c].nunique()>2 :
                        c2.append(c)
                        c3[c] = 'ohe_'+c

                df = pd.get_dummies(df,columns=c2,drop_first=True,prefix=c3)
                #print(df.shape)
                return df
            train1 = OHE(train)

            unwanted = ['msno','registration_init_time','transaction_date','membership_expire_date','date']

            train1.drop(unwanted,axis=1, inplace=True)

            # ----------- Categorization ----------- #

            cat_cols   = train1.nunique()[train1.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(train1.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in train1.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = train1.nunique()[train1.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            train1 = utils.preprocess(train1.copy(), num_cols, bin_cols, multi_cols, verbose=False)
            utils._write_dataframe_to_csv_on_s3(train1, path_out)

        elif dataset == "south-asian":

            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "south-asian/South Asian Wireless Telecom Operator (SATO 2015).csv"
            sep = ","
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            
            # data for writing file
            dataset = "SATO"

            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["churn"]



            # --------- Unique preprocessing for this data set --------- #

            # rename the target variable to Churn
            telcom.rename(columns={"Class":"churn"}, inplace=True)

            # change the "Churned" to 1
            telcom[target_col[0]].replace("Churned", "1", inplace=True)

            # change the active to 0
            telcom[target_col[0]].replace("Active", "0", inplace=True)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)
            utils._write_dataframe_to_csv_on_s3(telcom, path_out)
            
            

        elif dataset == "newspaper":
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "newspaper/NewspaperChurn.csv"
            sep = ","
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            
            # data for writing file
            dataset = "news"

            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["Churn"]



            # --------- Unique preprocessing for this data set --------- #

            # rename the target variable to Churn
            telcom.rename(columns={"Subscriber":"Churn"}, inplace=True)

            # change the "Churned" to 1
            telcom.Churn.replace("no", "0", inplace=True)

            # change the active to 0
            telcom.Churn.replace("yes", "1", inplace=True)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            del telcom['Address']
            del telcom["SubscriptionID"]
            del telcom['Zip Code']

            # useless all the obs. came from CA state
            del telcom['State']

            num_cols = ['Year Of Residence', 'reward program']

            multi_cols = ['weekly fee',
             'Age range',
             'City',
             'Source Channel',
             'County',
             'Ethnicity',
             'Deliveryperiod',
             'Nielsen Prizm',
             'HH Income',
             'Language']

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)
            utils._write_dataframe_to_csv_on_s3(telcom, path_out)


        elif dataset == "Kdd2009-small":
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "Kdd2009-small/orange_small_train.data"

            target_db_name = ["orange_small_train_churn.labels"]
            target_name = "Kdd2009-small/orange_small_train_churn.labels"
            sep = "\t"
            
            
            # data for writing file
            dataset = "K2009"
            
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["churn"]



            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep).replace('\\', '/')

            target = utils._read_dataframe_from_csv_on_s3(path, target_name, sep)

            target.columns = ["churn"]

            # merge training set with its target
            telcom = telcom.join(target)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]    

            # ----------- Preprocessing ----------- #

            # last observation is -0.788 for the churn so we remove it
            telcom = telcom[:-1]


            del telcom["churn"]

            # data cleaning: drop the columns with std close to zero
            # delete columns with at least 20% missing values
            threshold = 0.2
            telcom = telcom.drop(telcom.std()[telcom.std() < threshold].index.values, axis=1)
            telcom = telcom.loc[:, pd.notnull(telcom).sum() > len(telcom)*.8]

            DataVars = telcom.columns
            data_types = {Var: telcom[Var].dtype for Var in DataVars}

            for Var in DataVars:
                if data_types[Var] == int:
                    x = telcom[Var].astype(float)
                    telcom.loc[:, Var] = x
                    data_types[Var] = x.dtype
                elif data_types[Var] != float:
                    x = telcom[Var].astype('category')
                    telcom.loc[:, Var] = x
                    data_types[Var] = x.dtype        
            # numerical data
            float_DataVars = [Var for Var in DataVars
                                 if data_types[Var] == float]

            float_x_means = telcom.mean()

            for Var in float_DataVars:
                x = telcom[Var]
                isThereMissing = x.isnull()
                if isThereMissing.sum() > 0:
                    telcom.loc[isThereMissing.tolist(), Var] = float_x_means[Var]  
            DataVars = telcom.columns

            categorical_DataVars = [Var for Var in DataVars
                                       if data_types[Var] != float]

            categorical_levels = telcom[categorical_DataVars].apply(lambda col: len(col.cat.categories))

            categorical_DataVars = categorical_levels[categorical_levels <= 500].index

            col_to_keep = float_DataVars + list(categorical_DataVars)
            telcom = telcom[col_to_keep]

            collapsed_categories = {}

            removed_categorical_DataVars = set()

            for Vars in categorical_DataVars:

                isTheremissing_value = telcom[Vars].isnull()
                if isTheremissing_value.sum() > 0:
                    telcom[Vars].cat.add_categories('unknown', inplace=True)
                    telcom.loc[isTheremissing_value.tolist(), Vars] = 'unknown'

            cat_cols = telcom.select_dtypes("category").columns
            num_cols = telcom.select_dtypes("float").columns

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()

            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            # merge training set with its target
            telcom = telcom.join(target)

            # changing labels to 0 or 1
            telcom["churn"] = (telcom["churn"] +1)/2

            # convert to int the target variable othw it is 1.0 and 0.0
            telcom.churn = pd.to_numeric(telcom.churn, downcast='integer')
            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)
            utils._write_dataframe_to_csv_on_s3(telcom, path_out)


        elif dataset == "CrowdAnalytix":
            pass
        elif dataset == "TelcoCustChurn":
            path = "/churn_package/datasets"        

            # ------ input ------ #
            # Data file opening
            path = "/Churn/datasets"
            dataset_name = "TelcoCustChurn/Telco_Customer_Churn.csv"
            sep = ","

            # data preprocessing
            target_col = ["Churn"]

            # data for writing file
            dataset = "TelC"        
            # ------ writing scoring report ------ #
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"        
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            # --------- Unique preprocessing for this data set --------- #

            #Replacing blank spaces with null values in total charges column
            telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]
            #Dropping null values from total charges column which contain .15% missing data 
            #telcom = telcom[telcom["TotalCharges"].notnull()]
            #telcom = telcom.reset_index()[telcom.columns]

            # replace missing values by mean there are only 11 missing values
            telcom["TotalCharges"] = pd.to_numeric(telcom["TotalCharges"])
            telcom["TotalCharges"] = telcom["TotalCharges"].fillna(telcom["TotalCharges"].mean())


            #convert to float type
            #telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

            #telcom.fillna(telcom.mean(), inplace=True)

            #replace 'No internet service' to No for the following columns
            replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport','StreamingTV', 'StreamingMovies']
            for i in replace_cols : 
                telcom[i]  = telcom[i].replace({'No internet service' : 'No'})

            #replace values
            telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})

            #Tenure to categorical column
            def tenure_lab(telcom) :

                if telcom["tenure"] <= 12 :
                    return "Tenure_0-12"
                elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):
                    return "Tenure_12-24"
                elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48) :
                    return "Tenure_24-48"
                elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60) :
                    return "Tenure_48-60"
                elif telcom["tenure"] > 60 :
                    return "Tenure_gt_60"
            telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom),
                                                  axis = 1)

            #Drop tenure column
            #telcom = telcom.drop(columns = "tenure_group",axis = 1)

            Id_col     = ['customerID']
            target_col = ["Churn"]

            cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]
            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]        

            # --------- general preprocessing --------- #
            del telcom["customerID"]
            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)

            target_col = ["churn"]

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)


        elif dataset == 'Bank-data':                
            sep = ","
            target_col = ["Exited"]

            path = "/churn_package/datasets"
            dataset_name = "Bank-data/Churn_Modelling.csv"
            
            dataset = "Bank"

            # ------ writing scoring report ------ #
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)

            #Replacing blank spaces with null values in total charges column
            del telcom["RowNumber"]

            # Unknown in the dataset is a NaN
            telcom.replace('Unknown', np.nan, inplace=True)

            del telcom["CustomerId"]
            del telcom["Surname"]

            telcom.columns = telcom.columns.str.lower()

            # rename the target variable to Churn
            telcom.rename(columns={"exited":"churn"}, inplace=True)

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)

            multi_cols = ['NumOfProducts', 'Geography']
            num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)



        elif dataset == 'telecom-sigtel':                
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "telecom-sigtel/Telecom Churn Data SingTel.csv"
            sep = ","
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)

            # data preprocessing
            target_col = ["Churn"]

            # data for writing file
            dataset = "UCI"

            # ------ writing scoring report ------ #
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]


            del telcom["State"]
            del telcom["Phone Number"]

            multi_cols = ['Area Code']

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)        

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)



        elif dataset == 'UCI':     
            # on hold we do not use it anymore
            pass

        elif dataset == 'DSN-telecom-churn':                
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "DSN-telecom-churn/TRAIN.csv"
            sep = ","
            # ------ writing scoring report ------ #
            # data for writing file
            dataset = "DSN"
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["Churn Status"]



            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            


            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            del telcom['Customer ID']

            # works only on numerical data
            telcom.fillna(telcom.mean(), inplace=True)

            telcom['Network type subscription in Month 1'].fillna(telcom['Most Loved Competitor network in in Month 1'].mode().iloc[0], inplace=True)
            telcom['Network type subscription in Month 2'].fillna(telcom['Most Loved Competitor network in in Month 1'].mode().iloc[0], inplace=True)


            telcom['Most Loved Competitor network in in Month 1'].fillna(telcom['Most Loved Competitor network in in Month 1'].mode().iloc[0], inplace=True)
            telcom['Most Loved Competitor network in in Month 2'].fillna(telcom['Most Loved Competitor network in in Month 1'].mode().iloc[0], inplace=True)

            # remove the last obs. of the dataset, the churn value is "0.5" and it is the only one
            telcom["Churn Status"] = telcom.drop(telcom.index[-1])["Churn Status"]
            telcom = telcom[:-1]

            multi_cols = ['Most Loved Competitor network in in Month 2',
             'Network type subscription in Month 1',
             'Network type subscription in Month 2',
             'Most Loved Competitor network in in Month 1']

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)  
            
            telcom.rename(columns={"churn status":"churn"}, inplace=True)

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)



        elif dataset == 'membershipWoes':                
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "membershipWoes/Assignment- Membership woes.csv"
            sep = ","
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)
            # ------ writing scoring report ------ #
            # data for writing file
            dataset = "Member"  
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["Churn"]

      
            # --------- Unique preprocessing for this data set --------- #

            telcom.rename(columns = {'MEMBERSHIP_STATUS': "Churn"}, inplace=True)
            telcom["Churn"].map({'INFORCE': 0, 'CANCELLED': 1})

            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]        

            # preprocessing
            del telcom['START_DATE (YYYYMMDD)']
            del telcom['MEMBERSHIP_NUMBER']
            del telcom["AGENT_CODE"]
            del telcom['END_DATE  (YYYYMMDD)']

            num_cols = ['MEMBERSHIP_TERM_YEARS',
             'ANNUAL_FEES',
             'MEMBER_ANNUAL_INCOME',
             'MEMBER_AGE_AT_ISSUE']

            multi_cols =  ['ADDITIONAL_MEMBERS',
             'MEMBER_OCCUPATION_CD',
             'PAYMENT_MODE',
             'MEMBER_MARITAL_STATUS']

            # replace nan by most frequent value
            telcom["MEMBER_OCCUPATION_CD"].fillna(telcom["MEMBER_OCCUPATION_CD"].mode().iloc[0], inplace=True)
            telcom["MEMBER_GENDER"].fillna(telcom["MEMBER_GENDER"].mode().iloc[0], inplace=True)
            telcom["MEMBER_MARITAL_STATUS"].fillna(telcom["MEMBER_MARITAL_STATUS"].mode().iloc[0], inplace=True)

            # for numerical values replace nan by mean

            telcom.fillna(telcom.mean(), inplace=True)

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)

            utils._write_dataframe_to_csv_on_s3(telcom, path_out)

        elif dataset == 'IBM-HR': 
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "IBM-HR/WA_Fn-UseC_-HR-Employee-Attrition.csv"
            sep = ","
            # ------ writing scoring report ------ #
            # data for writing file
            dataset = "HR"
            
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)


            # data preprocessing
            target_col = ["Attrition"]
            
            telcom.rename(columns={"Attrition": "churn"}, inplace=True)


            # data for writing file
            dataset = "IBM-HR"

            # ----------- Categorization ----------- #
            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)
                        
            
            utils._write_dataframe_to_csv_on_s3(telcom, path_out)


        if dataset == "mobile-churn":
            # ------ input ------ #
            # Data file opening
            path = "/churn_package/datasets"
            dataset_name = "mobile-churn/mobile-churn-data.csv"
            sep = ","
            # ------ writing scoring report ------ #
            dataset = "Mobile"
            path_out = f"churn_package/churn/post_preprocessing_files/{dataset}.csv"

            # data preprocessing
            target_col = ["churn"]

            # data for writing file
            dataset = "mobile-churn"        

            # training dataset
            telcom = utils._read_dataframe_from_csv_on_s3(path, dataset_name, sep)        
            # ----------- Categorization ----------- #

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            del telcom["year"]
            del telcom["month"]
            del telcom["user_account_id"]

            cat_cols   = telcom.nunique()[telcom.nunique() < 10].keys().tolist()
            cat_cols   = [x for x in cat_cols if x not in target_col]

            cat_cols = list(set(cat_cols + list(telcom.select_dtypes(include=["object"]).columns)))

            num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col]

            #Binary columns with 2 values
            bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
            #Columns more than 2 values
            multi_cols = [i for i in cat_cols if i not in bin_cols]

            telcom[multi_cols] = telcom[multi_cols].apply(lambda x: x.str.replace(',','.'))

            telcom[multi_cols] = telcom[multi_cols].apply(pd.to_numeric)

            num_cols = num_cols + multi_cols

            multi_cols = []
            # une fois que j'ai changer la virgule en point, j'ai eu masse NaN bizarre
            telcom = telcom.fillna(telcom.mean())

            telcom = utils.preprocess(telcom.copy(), num_cols, bin_cols, multi_cols, verbose=False)



            utils._write_dataframe_to_csv_on_s3(telcom, path_out)        