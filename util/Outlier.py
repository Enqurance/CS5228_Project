import pandas as pd


# Handle outlier use 3 sigma
# Identify outliers based on the mean and standard deviation of the data.
def handle_outliers_3sigma(df, name):
	# mean and std
	mean = df[name].mean()
	standard = df[name].std()

	# 3-sigma
	lower_bound = mean - 3 * standard
	upper_bound = mean + 3 * standard

	outliers = df[(df[name] < lower_bound) | (df[name] > upper_bound)]

	print("Number of ", name, "outliers:", outliers.shape[0])
	print("List of ", name, "outliers: ", outliers[name].tolist())

	df_cleaned = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]

	return df_cleaned


# Handle outlier use Interquartile Range
# Identify and handle outliers using the quartiles of the data.
def handle_outliers_iqr(df, name):
	Q1 = df[name].quantile(0.25)
	Q3 = df[name].quantile(0.75)
	IQR = Q3 - Q1

	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR

	outliers = df[(df[name] < lower_bound) | (df[name] > upper_bound)]

	print(f"Number of outliers: {outliers.shape[0]}")
	print(f"List of outliers: {outliers[name].tolist()}")

	df_cleaned = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]

	return df_cleaned


# Handle outlier use percentile
# Values in the lowest 1% or the highest 1% are considered outliers and should be removed.
def handle_outliers_percentile(df, name):
	lowpercentile = df[name].quantile(0.01)
	uppercentile = df[name].quantile(0.99)

	outliers = df[(df[name] < lowpercentile) | (df[name] > uppercentile)]

	print(f"Number of outliers: {outliers.shape[0]}")
	print(f"List of outliers: {outliers[name].tolist()}")

	df_cleaned = df[(df[name] >= lowpercentile) & (df[name] <= uppercentile)]

	return df_cleaned


# This function apply 3-sigma law to remove outliers
def remove_outliers_by_group(df, group_column, target_column):
	def remove_outliers_3sigma(group):
		mean = group[target_column].mean()
		std = group[target_column].std()

		lower_limit = mean - 3 * std
		upper_limit = mean + 3 * std

		return group[(group[target_column] >= lower_limit) & (group[target_column] <= upper_limit)]

	df_filtered = df.groupby(group_column).apply(remove_outliers_3sigma)
	df_filtered = df_filtered.reset_index(drop=True)

	return df_filtered
