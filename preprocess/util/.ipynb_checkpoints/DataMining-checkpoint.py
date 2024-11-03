from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


def RandomForestMiningByModel(x_train, x_test, y_train, y_test):
	# Normalize the attributes
	model_dict = {}
	y_pred = None
	for model_type, group in x_train.groupby('model'):
		X = group.drop(['model'], axis=1)
		y = y_train[y_train['model'] == model_type].drop(['model'], axis=1)

		model = RandomForestRegressor(n_estimators=100, random_state=42)

		model.fit(X, np.ravel(y))
		model_dict[model_type] = model

	for model_type, group in x_test.groupby('model'):
		X = group.drop(['model'], axis=1)

		if model_type not in model_dict.keys():
			y_test = y_test[y_test['model'] != model_type]
			continue

		model = model_dict[model_type]
		y_pred_new = model.predict(X)

		# 将预测结果与索引结合为 DataFrame
		y_pred_new_df = pd.DataFrame(y_pred_new, index=X.index, columns=['prediction'])

		# 将新的预测结果与之前的结果合并
		if y_pred is None:
			y_pred = y_pred_new_df
		else:
			y_pred = pd.concat([y_pred, y_pred_new_df])

	y_test = y_test.drop(['model'], axis=1)
	y_test_aligned = y_test.reindex(y_pred.index)
	rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
	print(f'RMSE on test data: {rmse}')

	return rmse


def RandomForestMining(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	model = RandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	# Calculate RMSE
    print(dev)
    # if dev:
    #     return y_pred
    # else:
    #     print('Not developing')
    #     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #     print(f'RMSE on test data: {rmse}')
    #     return rmse


def SVRMining(x_train, x_test, y_train, y_test):
	print("Using SVR for data mining")
	scaler = StandardScaler()

	# Scaling
	x_trian = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	# Load SVR model
	svr = SVR(kernel='rbf', C=100, gamma='auto')
	svr.fit(x_trian, y_train)

	# Train and test
	y_pred = svr.predict(x_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print(f'Root Mean Squared Error: {rmse}')


def KNNMining(x_train, x_test, y_train, y_test):
	# 'Price' is the target
	print("Using KNN for data mining")

	# Load KNN model
	knn_model = KNeighborsRegressor(n_neighbors=100)

	# Cross validation
	cv_scores = cross_val_score(knn_model, x_train, y_train, cv=20, scoring='neg_mean_squared_error')
	mean_cv_score = np.sqrt(-cv_scores.mean())
	print(f'Root Mean Cross-Validated MSE: {mean_cv_score}')

	# Train and test
	knn_model.fit(x_train, y_train)
	y_pred = knn_model.predict(x_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print(f'Root Mean Squared Error on Test Set: {rmse}')


def split_dataframe(df, target_cols, test_size=0.2, random_state=None):
	X = df.drop(columns=target_cols)
	y = df[target_cols]
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	return x_train, x_test, y_train, y_test


def split_dataframe_flex(df, train_drop_cols, test_cols, test_size=0.2, random_state=None):
	X = df.drop(columns=train_drop_cols)
	y = df[test_cols]
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	return x_train, x_test, y_train, y_test
