from azureml.core import Run
import pandas as pd
from argparse import ArgumentParser as AP
from sklearn.preprocessing import MinMaxScaler


"""
Get the run context 
Recall : we are in the sub-script (submited by the pipeline job)
"""
new_run = Run.get_context()
"""
###############################################################################
"""


"""
Get the workspace from the run 
"""
ws = new_run.experiment.workspace
"""
###############################################################################
"""


"""
Preprocessing data 
"""
# read input data
df = new_run.input_dataset["raw_data"].to_pandas_dataframe 
#df = pd.read_csv("./data/Loan+Approval+Prediction.csv")
# remove ID coolumn (dont need Id column for build pipeline)
data_preproc = df.drop(["ID"], axis=1)

#get columns of dataframe
all_cols = data_preproc.columns

# get missing values as vector of sum
dataNull = data_preproc.isnull().sum()

# replace missing values of string variables (columns) with mode method
mode = data_preproc.mode().iloc[0]
cols = data_preproc.select_dtypes(include="object").columns
data_preproc[cols] = data_preproc[cols].fillna(mode)

# replace missing values of string variables (columns) with mean
mean = data_preproc.mean()
data_preproc = data_preproc.fillna(mean)

# create dummy variables (1-hot encoding)
data_preproc = pd.get_dummies(data_preproc, drop_first=True)

# normalize data
scaler = MinMaxScaler()
columns = df.select_dtypes(include="number").columns
data_preproc[columns] = scaler.fit_transform(data_preproc[columns])

# get the arguments from the pipline job
parser = AP()
parser.add_argument("--datafolder", type=str)
args = parser.parse_args()

# create folder if does not exist
import os
os.makedev(args.datafolder, exist_ok=True)

# create the path
path = os.path.join(args.datafolder, "defaults_prep.csv")

# write the data_preproc ad csv file
data_preproc.to_csv(path, index=False)
"""
###############################################################################
"""


"""
Complete the run 
"""
new_run.complete()
"""
###############################################################################
"""