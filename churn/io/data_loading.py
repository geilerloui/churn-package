import s3fs
import boto3
from io import StringIO
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# to display full width for top3variables
pd.options.display.max_colwidth = 100

# ------ homemade .py ------ #
s3 = boto3.resource('s3')
s3.meta.client.download_file(os.environ["S3_BUCKET"], "churn_package/churn/utils.py", "utils.py")

import utils

