from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

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


def ScalarWithoutMakeModel(x_train, x_test):
	scaler = StandardScaler()
	columns_to_scale = x_train.columns.tolist()
	columns_to_scale = [col for col in columns_to_scale if col not in ['make', 'model']]

	# Fit the scaler on the training data
	x_train_scaled_part = scaler.fit_transform(x_train[columns_to_scale])
	x_train_scaled_part_df = pd.DataFrame(x_train_scaled_part, columns=columns_to_scale)

	# Transform the test data using the same scaler
	x_test_scaled_part = scaler.transform(x_test[columns_to_scale])
	x_test_scaled_part_df = pd.DataFrame(x_test_scaled_part, columns=columns_to_scale)

	# Create new scaled DataFrames
	x_train_scaled_df = x_train.copy()
	x_train_scaled_df[columns_to_scale] = x_train_scaled_part_df

	x_test_scaled_df = x_test.copy()
	x_test_scaled_df[columns_to_scale] = x_test_scaled_part_df

	return x_train_scaled_df, x_test_scaled_df


def XGBoostMining(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)
	model = xgb.XGBRegressor(
		n_estimators=1500,
		learning_rate=0.05,
		max_depth=4,
		subsample=1,
		random_state=42
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


def XGBoostMiningByMake(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	y_train = y_train
	x_train, x_test = ScalarWithoutMakeModel(x_train, x_test)

	model_complete = xgb.XGBRegressor(
		n_estimators=1500,
		learning_rate=0.05,
		max_depth=4,
		subsample=1,
		random_state=42
	)
	model_complete.fit(x_train, y_train.drop(columns=['make']))

	model_dict = {}
	for car_make, group in x_train.groupby('make'):
		X = group.drop(['make'], axis=1)
		y = y_train[y_train['make'] == car_make].drop(['make'], axis=1)
		n_estimators = len(group) * 5
		model = xgb.XGBRegressor(
			n_estimators=n_estimators,
			learning_rate=0.05,
			max_depth=4,
			subsample=1,
			random_state=42
		)

		model.fit(X, np.ravel(y))
		model_dict[car_make] = model

	y_pred = pd.DataFrame(index=x_test.index, columns=['make', 'Predicted'])
	missing_models = pd.DataFrame(columns=x_train.columns)

	for index, row in x_test.iterrows():
		if row['make'] in model_dict:
			model = model_dict[row['make']]
			X = row.drop(['make']).to_frame().T
			y_pred_new = model.predict(X)
			new_row = pd.DataFrame({'make': [row['make']], 'Predicted': [y_pred_new[0]]}, index=[index])
			y_pred.loc[index] = new_row.iloc[0]
		else:
			row_df = pd.DataFrame([row], index=[index])
			missing_models = pd.concat([missing_models, row_df])

	if not missing_models.empty:
		X_missing = missing_models
		y_missing_pred = model_complete.predict(X_missing)
		y_missing_pred_df = pd.DataFrame(y_missing_pred, columns=['Predicted'], index=missing_models.index)
		result_df = pd.concat([missing_models['make'], y_missing_pred_df], axis=1)
		y_pred.update(result_df)

	y_pred.reset_index(inplace=True)
	y_pred = y_pred.drop(columns=['make'])
	y_pred.rename(columns={'index': 'Id'}, inplace=True)


	# Calculate RMSE
	if dev:

		return y_pred
	else:
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_test['price'], y_pred['Predicted']))
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
			max_depth=4
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
		return y_pred[['Predicted']]
	else:
		y_pred['Actual'] = y_test['price']
		y_pred.to_csv('./data/results.csv', index=False)
		print("Data saved to results.csv")
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_pred['Predicted'], y_test['price']))
		print(f'RMSE on test data: {rmse}')
		return rmse


def GradientBoostingMiningByModel(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	model_dict = {}
	for model_type, group in x_train.groupby('model'):
		X = group.drop(['model'], axis=1)
		y = y_train[y_train['model'] == model_type].drop(['model'], axis=1)

		model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=15, random_state=42)

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
		y_pred = y_pred.reindex(x_test.index)
		return y_pred[['Predicted']]
	else:
		y_pred['Actual'] = y_test['price']
		y_pred.to_csv('./data/results.csv', index=False)
		print("Data saved to results.csv")
		print('Running not in develop mode')
		rmse = np.sqrt(mean_squared_error(y_pred['Predicted'], y_test['price']))
		print(f'RMSE on test data: {rmse}')
		return rmse


def GradientBoostingMining(x_train, x_test, y_train, y_test=None, dev=False):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)
	model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=42)
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
