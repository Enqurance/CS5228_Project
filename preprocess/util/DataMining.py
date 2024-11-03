from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

import numpy as np
import pandas as pd


def CombinedDataMiningRandomForestAndLinearRegression(df, df_test):
	df_dereg_nan = df_test[df_test['dereg_value'].isna()].reset_index(drop=False)
	df_dereg = df_test[df_test['dereg_value'].notna()].reset_index(drop=False)
	# For rows with dereg_value, we apply linear regression here
	x_train = df[['dereg_value', 'model']]
	y_train = df[['price', 'model']]
	x_test = df_dereg[['dereg_value', 'model']]
	pred_dereg = LinearRegressionMiningByModel(x_train, x_test, y_train, dev=True)

	# For rows without dereg_value, we apply random forest here
	x_train = df.drop(columns=['price', 'dereg_value', 'depreciation'])
	y_train = df['price']
	x_test = df_dereg_nan.drop(columns=['dereg_value', 'depreciation', 'index'])
	pred_dereg_nan = RandomForestMining(x_train, x_test, y_train, dev=True)
	pred_dereg['index'] = df_dereg['index'].values
	pred_dereg_nan['index'] = df_dereg_nan['index'].values

	combined_pred = pd.concat([pred_dereg, pred_dereg_nan])
	combined_pred = combined_pred.sort_values(by='index').reset_index(drop=True)
	combined_pred['Id'] = range(len(combined_pred))
	combined_pred = combined_pred.drop(columns=['index'])

	return combined_pred


def LinearRegressionMiningByModel(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	model_dict = {}
	y_pred = None
	for model_type, group in x_train.groupby('model'):
		X = group.drop(['model'], axis=1)
		y = y_train[y_train['model'] == model_type].drop(['model'], axis=1)

		model = LinearRegression()
		model.fit(X, y)
		model_dict[model_type] = model

	y_pred_df = pd.DataFrame(columns=['model', 'prediction'])
	for test in x_test:
		model = model_dict[test['model']]
		X = test.drop(['model'], axis=1)
		y_pred_new = model.predict(X)

		temp_df = pd.DataFrame({
			'model': [test['model']] * len(y_pred_new),
			'prediction': y_pred_new
		})

		y_pred_df = pd.concat([y_pred_df, temp_df], ignore_index=True)

	y_pred_aligned = y_pred.loc[x_test.index]
	if dev:
		return y_pred_aligned
	else:
		y_test = y_test.drop(['model'], axis=1)
		y_test_aligned = y_test.reindex(y_pred_aligned.index)
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_test_aligned, y_test))
		print(f'RMSE on test data: {rmse}')
		return rmse


def LinearRegressionMining(x_train, x_test, y_train, y_test=None, dev=False):
	model = LinearRegression()
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	# Calculate RMSE
	if dev:
		ids = [i for i in range(len(x_test))]
		return pd.DataFrame(
			list(zip(ids, y_pred)),
			columns=['Id', 'Predicted']
		)
	else:
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		print(f'RMSE on test data: {rmse}')
		return rmse


def RandomForestMiningByModel(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	model_dict = {}
	for model_type, group in x_train.groupby('model'):
		X = group.drop(['model'], axis=1)
		y = y_train[y_train['model'] == model_type].drop(['model'], axis=1)

		model = RandomForestRegressor(
			n_estimators=200,
			max_depth=16
		)

		model.fit(X, np.ravel(y))
		model_dict[model_type] = model

	y_pred = pd.DataFrame(columns=['model', 'Predicted'])
	for index, test in x_test.iterrows():
		model = model_dict[test['model']]
		X = test.drop(['model']).to_frame().T
		y_pred_new = model.predict(X)

		temp_df = pd.DataFrame({
			'model': [test['model']] * len(y_pred_new),
			'Predicted': y_pred_new,
		}, index=[index])  # Use the original index for temp_df
		y_pred = pd.concat([y_pred, temp_df])

	if dev:
		# Reindex y_pred to match x_test's index
		y_pred = y_pred.reindex(x_test.index)
		return y_pred['Predicted']
	else:
		y_pred['Actual'] = y_test['price']
		y_pred.to_csv('./data/results.csv', index=False)
		print("Data saved to results.csv")
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_pred['Predicted'], y_test['price']))
		print(f'RMSE on test data: {rmse}')
		return rmse


def GradientBoostingMiningByModel(x_train, x_test, y_train, y_test=None, dev=False):
	model_dict = {}

	for model_type, group in x_train.groupby('model'):
		X = group.drop(['model'], axis=1)
		y = y_train[y_train['model'] == model_type].drop(['model'], axis=1)

		model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=15, random_state=42)

		if dev:
			cv = KFold(n_splits=5, shuffle=True, random_state=42)
			cv_scores = cross_val_score(model, X, np.ravel(y), cv=cv, scoring='neg_mean_squared_error')
			cv_rmse = np.sqrt(-cv_scores)
			print(f'Model {model_type} - Cross-validation RMSE scores: {cv_rmse}')
			print(f'Model {model_type} - Average CV RMSE: {cv_rmse.mean()}')

		model.fit(X, np.ravel(y))
		model_dict[model_type] = model

	y_pred = pd.DataFrame(columns=['model', 'Predicted'])

	for index, test in x_test.iterrows():
		model = model_dict[test['model']]
		X = test.drop(['model']).to_frame().T
		y_pred_new = model.predict(X)

		temp_df = pd.DataFrame({
			'model': [test['model']] * len(y_pred_new),
			'Predicted': y_pred_new,
		}, index=[index])
		y_pred = pd.concat([y_pred, temp_df])

	if dev:
		y_pred = y_pred.reindex(x_test.index)
		return y_pred['Predicted']
	else:
		y_pred['Actual'] = y_test['price']
		y_pred.to_csv('./data/results.csv', index=False)
		print("Data saved to results.csv")
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_pred['Predicted'], y_test['price']))
		print(f'RMSE on test data: {rmse}')
		return rmse


def GradientBoostingMining(x_train, x_test, y_train, y_test=None, dev=True):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	# Initialize the model
	model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)

	# Perform cross-validation if in development mode
	if not dev:
		cv = KFold(n_splits=5, shuffle=True, random_state=42)
		cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
		cv_rmse = np.sqrt(-cv_scores)
		print(f'Cross-validation RMSE scores: {cv_rmse}')
		print(f'Average CV RMSE: {cv_rmse.mean()}')

	# Fit the model on the training data
	model.fit(x_train_scaled, y_train)

	# Predict on the test data
	y_pred = model.predict(x_test_scaled)
	y_pred_df = pd.DataFrame(y_pred, index=x_test.index, columns=['Predicted'])

	# Calculate RMSE if y_test is provided
	if y_test is not None:
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		print(f'RMSE on test data: {rmse}')
		return y_pred_df, rmse

	return y_pred_df


def RandomForestMining(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)
	model = RandomForestRegressor(
		n_estimators=500,
		max_depth=16
	)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test_scaled)
	y_pred = pd.DataFrame(y_pred, index=x_test.index, columns=['Predicted'])

	# Calculate RMSE
	if dev:
		return y_pred
	else:
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		print(f'RMSE on test data: {rmse}')
		return rmse


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
