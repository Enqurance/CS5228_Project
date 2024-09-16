from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


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
