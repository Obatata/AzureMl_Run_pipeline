from azureml.core import Run
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
Get the parameters 
"""
parser = argparse.ArgumentParser()
parser.add_argument("--datafolder", type=str)
args = parser.parse_args()
"""
###############################################################################
"""


"""
Training process 

Important !!! 
defaults_prep.csv : is the data provided from the previous step (preprocessig)
"""
# get data from the previous step (prepocessing data step)
path_data = os.path.join(args.datafolder, "defaults_prep.csv")
data_prep = pd.read_csv(path_data)
# create X (features) and Y (target)
y = data_prep[['Loan_Status_Y']]
X = data_prep.drop(['Loan_Status_Y'], axis=1)
# Split data X and Y into test and training datasets
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y,
                                                    stratify=y,
                                                    random_state=123,
                                                    test_size=0.3
                                                    )
# build logistic regression model
lr = LogisticRegression()
# fit the logistic regression over the data
lr.fit(X_train, y_train)
# test data prediction
y_pred = lr.predict(X_test)
# get probabilities prediction
y_prob = lr.predict_proba(X)[:, 1]
# model evaluation (confusio metrix and accuracy score)
confusion_matrix = confusion_matrix(y_test, y_pred)
score = lr.score(X_test, y_test)
"""
###############################################################################
"""


"""
Log metrics 
"""
# create confusion metrics dictionnary
confusion_matrix_dict = {
                         "schema_type":"confusion_matrix",
                         "schema_version":"v1",
                         "data":{
                                "class_labels": ["N", "Y"],
                                "matrix":confusion_matrix.tolist()
                                }
                        }

new_run.log_confusion_matrix("ConfusionMatrix", confusion_matrix_dict)
new_run.log("ScoreAccuracy", score)
"""
##############################################################################
"""


"""
Create scored dataset and upload  to outputs
"""
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

y_prob_df    = pd.DataFrame(y_prob, columns=["Scored Probabilities"])
y_predict_df = pd.DataFrame(y_pred, columns=["Scored Label"])

scored_dataset = pd.concat([X_test, y_test, y_predict_df, y_prob_df],
                           axis=1)

scored_dataset.to_csv(
                      "./outputs/scored_dataset.csv",
                      sep=";",
                      index=False
                    )
"""
#############################################################################
"""


"""
Complete the run 
"""
new_run.complete()
"""
#############################################################################
"""
