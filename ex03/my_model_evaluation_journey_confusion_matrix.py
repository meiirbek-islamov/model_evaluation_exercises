import pandas as pd 
import numpy as numpy
from sklearn.metrics import confusion_matrix
def my_model_evaluation_journey_confusion_matrix(param_1, param_2):
    # data_1 = pd.read_csv(param_1, ",").select_dtypes(include=['float64', 'int64'])
    # data_2 = pd.read_csv(param_2, ",").select_dtypes(include=['float64', 'int64'])
    return confusion_matrix(param_1, param_2)