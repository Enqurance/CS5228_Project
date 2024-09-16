import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def KNNMining(df):
	# 假设你的数据存储在一个名为 df 的 DataFrame 中
	# 并且 'price' 是你要预测的目标变量
	# 选择特征和目标变量
	X = df.drop('price', axis=1)  # 假设 'price' 是目标变量
	y = df['price']

	# 数据标准化
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# 将数据分为训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

	# 创建 KNN 回归模型
	knn_model = KNeighborsRegressor(n_neighbors=50)

	# 使用交叉验证评估模型
	cv_scores = cross_val_score(knn_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
	mean_cv_score = -cv_scores.mean()
	print(f'Mean Cross-Validated MSE: {mean_cv_score}')

	# 在整个训练集上训练模型并评估测试集
	knn_model.fit(X_train, y_train)
	y_pred = knn_model.predict(X_test)
	print(f'Mean Squared Error on Test Set: {mse}')
