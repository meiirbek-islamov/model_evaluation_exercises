# import pip
# pip.main(['install', 'sklearn'])
import pandas as pd 
import numpy as numpy
from sklearn.metrics import r2_score

def my_model_evaluation_journey_r2_score(param_1, param_2):
    data_1 = pd.read_csv(param_1, ",").select_dtypes(include=['float64', 'int64'])
    data_2 = pd.read_csv(param_2, ",").select_dtypes(include=['float64', 'int64'])
    return r2_score(data_1.iloc[range(len(data_1))], data_2.iloc[range(len(data_1))])

# print(my_model_evaluation_journey_r2_score("test.csv", "test1.csv"))
