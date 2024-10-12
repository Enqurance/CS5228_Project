from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd

from util.Outlier import remove_outliers_by_group

columns_to_delete = [
	'listing_id',
	'title',
	'description',
	'original_reg_date',
	'fuel_type',  # We may consider filling up this column later
	'opc_scheme',
	'lifespan',
	'eco_category',
	'features',
	'accessories',
	'indicative_price'
]


def ReadCSV(file_path):
	with open(file_path, 'r') as file:
		df = pd.read_csv(file)

	return df


def WriteCSV(df, file_path):
	df.to_csv(file_path, index=False)


def PreprocessData(in_path, out_path):
	df = ReadCSV(in_path)

	# Data cleaning
	# Dealing with missing values
	df = HandlingMissingValues(df)
	total_nan_count = df.isna().sum().sum()
	print("Missing values after cleaning:", total_nan_count)

	# Removing useless columns
	df = DeleteColumns(df)

	# Data encoding
	df = HandlingCategoryAttribute(df)
	df = DataEncoding(df)

	# Data
	WriteCSV(df, out_path)


# Columns with NaN values：['make', 'description', 'manufactured', 'original_reg_date', 'curb_weight', 'power',
# 				'fuel_type', 'engine_cap', 'no_of_owners', 'depreciation', 'road_tax', 'dereg_value', 'mileage',
# 				'omv', 'arf', 'opc_scheme', 'lifespan', 'features', 'accessories', 'indicative_price']
def HandlingMissingValues(df):
	# Step: fill in the missing values in column 'make' since we can infer them
	# Fill in the column 'make' base on the column 'make' and 'model'
	# This part of code reduce the null value to 0 for column 'make'
	model_dict = {}
	for index, row in df.iterrows():
		if row['model'] not in model_dict.keys() or pd.isna(model_dict[row['model']]):
			model_dict[row['model']] = row['make']

	for index, row in df.iterrows():
		if pd.isna(row['make']):
			df.loc[index, 'make'] = model_dict[row['model']]

	# Step: remove the rows with the 'manufactured' column missing, 7 rows in total
	df = df.dropna(subset=['manufactured'])

	# Step: fill in the missing values in column 'power'
	# The power of cars for a certain model are very likely to be similar
	# So we take the average values of power of each car model
	# If there are still missing values, we take the average of 'type_of_vehicle'
	mean_values = df.groupby('model')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	mean_values = df.groupby('type_of_vehicle')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	# Step: fill in the missing values in column 'engine_cap'
	# We do the same as we did in step 3 here
	mean_values = df.groupby('model')['engine_cap'].transform('mean')
	df.loc[:, 'engine_cap'] = df['engine_cap'].fillna(mean_values)
	df.loc[:, 'engine_cap'] = df['engine_cap'].round()

	mean_values = df.groupby('type_of_vehicle')['engine_cap'].transform('mean')
	df.loc[:, 'engine_cap'] = df['engine_cap'].fillna(mean_values)
	df.loc[:, 'engine_cap'] = df['engine_cap'].round()

	# Step: fill in the missing values in column 'no_of_owners'
	# We use the global average here
	mean_values = df['no_of_owners'].mean()
	df.loc[:, 'no_of_owners'] = df['no_of_owners'].fillna(mean_values)
	df.loc[:, 'no_of_owners'] = df['no_of_owners'].round()

	# Step: we remove rows where 'depreciation' or 'dereg_value' is null gere
	# Around 24,400 rows left after this step
	df = df.dropna(subset=['depreciation', 'dereg_value'])

	# Step: we handle road_tax NaN values, around 2,600 in total.
	# First, we fill up NaN values w.r.t. engine_cap
	road_tax_dict = {}
	for index, row in df.iterrows():
		if row['engine_cap'] not in road_tax_dict.keys() or pd.isna(road_tax_dict[row['engine_cap']]):
			road_tax_dict[row['engine_cap']] = row['road_tax']

	for index, row in df.iterrows():
		if pd.isna(row['road_tax']):
			df.loc[index, 'road_tax'] = road_tax_dict[row['engine_cap']]

	# After filling up with road_tax w.r.t. engine_cap, there are still around 1,000 NaN values
	# We use linear approximation to fill up these values after using EDA
	df_tmp = df.dropna(subset=['road_tax'])

	x = df_tmp['engine_cap'].values.reshape(-1, 1)
	y = df_tmp['road_tax'].values
	model = LinearRegression()
	model.fit(x, y)

	# Find indices with missing road_tax values and predict
	missing_indices = df[df['road_tax'].isnull()].index
	x_missing = df.loc[missing_indices, 'engine_cap'].values.reshape(-1, 1)
	y_pred = model.predict(x_missing)
	df.loc[missing_indices, 'road_tax'] = [round(yi) for yi in y_pred]

	# Step: we handle missing values in column 'mileage' here
	# We do random filling here
	missing_indices = df['mileage'].isnull()
	num_missing = missing_indices.sum()

	random_values = np.random.randint(1000, 200001, size=num_missing)
	df.loc[missing_indices, 'mileage'] = random_values

	# Step: we handle missing values in column 'omv' and 'arf' here
	# Since there are only around 100 missing values in total in this two columns
	# We directly drop data points with these columns empty
	df = df.dropna(subset=['omv', 'arf'])

	total_nulls = df.isnull().sum().sum()
	print("NaN values after handling: ", total_nulls)
	return df


def HandlingCategoryAttribute(df):
	# Replace '-' with an empty string
	df['category'] = df['category'].replace('-', '')

	# Split the 'category' column into lists
	df['category_list'] = df['category'].str.split(', ')

	# Handle empty strings by replacing them with empty lists
	df['category_list'] = df['category_list'].apply(lambda x: [] if x == [''] else x)

	# Import itertools for flattening lists
	from itertools import chain

	# Flatten the list of lists to a single list
	all_categories = list(chain.from_iterable(df['category_list']))

	# Get the unique categories
	unique_categories = set(all_categories)

	# Print the number of unique categories
	print(f"Number of unique categories: {len(unique_categories)}")
	print("Unique categories:", unique_categories)

	# Initialize the MultiLabelBinarizer
	mlb = MultiLabelBinarizer()

	# Fit and transform the category lists
	category_dummies = mlb.fit_transform(df['category_list'])

	# Create a DataFrame with the one-hot encoded categories
	category_df = pd.DataFrame(category_dummies, columns=mlb.classes_, index=df.index)

	# Concatenate the new dummy columns to the original DataFrame
	df = pd.concat([df, category_df], axis=1)

	# Drop the temporary 'category_list' column if desired
	df.drop('category_list', axis=1, inplace=True)
	df.drop('category', axis=1, inplace=True)

	num_records, num_attributes = df.shape

	print("There are {} data points, each with {} attributes.".format(num_records, num_attributes))
	return df


# We apply data encoding here
def DataEncoding(df):
	# We handle the attribute 'category' here
	# df = HandlingCategoryAttribute(df)
	return df


# Remove outlier by group
def OutlierRemoval(df, group_column, target_column):
	# For column omv, we apply 3-sigma law to remove outliers by group
	df = remove_outliers_by_group(df, group_column, target_column)

	num_records, num_attributes = df.shape
	print("There are {} data points, each with {} attributes".format(num_records, num_attributes))
	return df


def DeleteColumns(df):
	return df.drop(columns=columns_to_delete)
