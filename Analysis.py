import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

with open('./data/results.csv') as csvfile:
	df = pd.read_csv(csvfile)


def calculate_rmse(group):
	actual = group['Actual']
	predicted = group['Predicted']
	rmse = np.sqrt(mean_squared_error(actual, predicted))
	return rmse


rmse_by_group = df.groupby('model', group_keys=False).apply(calculate_rmse)

print(rmse_by_group.head())
rmse_by_group.to_csv('./temp_data/rmse_by_group_w_augmentation.csv')

