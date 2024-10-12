# This function apply 3-sigma law to remove outliers
def remove_outliers_by_group(df, group_column, target_column):
	def remove_outliers_3sigma(group):
		mean = group[target_column].mean()
		std = group[target_column].std()

		lower_limit = mean - 3 * std
		upper_limit = mean + 3 * std

		return group[(group[target_column] >= lower_limit) & (group[target_column] <= upper_limit)]

	def remove_outliers_percentile(name):
		lowpercentile = df[name].quantile(0.05)
		uppercentile = df[name].quantile(0.95)

		outliers = df[(df[name] < lowpercentile) | (df[name] > uppercentile)]

		print(f"Number of outliers: {outliers.shape[0]}")
		print(f"List of outliers: {outliers[name].tolist()}")

		df_cleaned = df[(df[name] >= lowpercentile) & (df[name] <= uppercentile)]

		return df_cleaned

	def handle_outliers_iqr(name):
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

	df_filtered = df.groupby(group_column).apply(remove_outliers_3sigma)
	df_filtered = df_filtered.reset_index(drop=True)

	return df_filtered
