from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


def RandomForest(x_train, x_test, y_train, y_test):
	# Normalize the attributes
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	model = RandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	# Calculate RMSE
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print(f'RMSE on test data: {rmse}')


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
