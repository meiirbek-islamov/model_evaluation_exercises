import pandas as pd 
import numpy as numpy
from sklearn.metrics import accuracy_score
def my_model_evaluation_journey_accuracy_score(param_1, param_2):
    # data_1 = pd.read_csv(param_1, ",").select_dtypes(include=['float64', 'int64'])
    # data_2 = pd.read_csv(param_2, ",").select_dtypes(include=['float64', 'int64'])
    return accuracy_score(param_1, param_2)