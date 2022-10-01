import pandas as pd 
import numpy as numpy
from sklearn.metrics import mean_squared_error
def my_model_evaluation_journey_mean_squared_error(param_1, param_2):
    data_1 = pd.read_csv(param_1, ",").select_dtypes(include=['float64', 'int64'])
    data_2 = pd.read_csv(param_2, ",").select_dtypes(include=['float64', 'int64'])
    return mean_squared_error(data_1.iloc[range(len(data_1))], data_2.iloc[range(len(data_1))])