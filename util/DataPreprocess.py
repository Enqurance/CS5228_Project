from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
from datetime import datetime

from util.Outlier import remove_outliers_by_group
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

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


def CalculateCarAge(df):
	if 'reg_date' in df.columns:
		df['reg_date'] = pd.to_datetime(df['reg_date'], format='%d-%b-%Y')
		df['reg_year'] = df['reg_date'].dt.year
		df = df.drop(columns=['reg_date'])
		current_year = datetime.now().year
		df['car_age'] = current_year - df['reg_year']
		df.drop(columns=['reg_year'])
		df.drop(columns=['manufactured'])

	return df


# Columns with NaN values：['make', 'description', 'manufactured', 'original_reg_date', 'curb_weight', 'power',
# 				'fuel_type', 'engine_cap', 'no_of_owners', 'depreciation', 'road_tax', 'dereg_value', 'mileage',
# 				'omv', 'arf', 'opc_scheme', 'lifespan', 'features', 'accessories', 'indicative_price']

def DataCalculation(df):
	if 'omv' in df.columns and 'arf' in df.columns:
		if (df['arf'] == 0).any():
			print("Warning: 'arf' column contains zero values. Division by zero will result in NaN.")

		df['omv_arf_ratio'] = df['omv'] / df['arf']

	if 'dereg_value' in df.columns and 'coe' in df.columns:
		if (df['coe'] == 0).any():
			print("Warning: 'arf' column contains zero values. Division by zero will result in NaN.")

		df['dereg_coe_ratio'] = df['dereg_value'] / df['coe']

	return df


def HandlingMissingValue(df):
	# Step: fill in the missing values in column 'power'
	# The power of cars for a certain model are very likely to be similar
	# So we take the average values of power of each car model
	# If there are still missing values, we take the average of 'type_of_vehicle'
	mean_values = df.groupby('model')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	mean_values = df.groupby('make')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	mean_values = df.groupby('type_of_vehicle')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	# # Step: fill in the missing values in column 'engine_cap'
	# # We do the same as we did in step 3 here
	# mean_values = df.groupby('model')['engine_cap'].transform('mean')
	# df.loc[:, 'engine_cap'] = df['engine_cap'].fillna(mean_values)
	# df.loc[:, 'engine_cap'] = df['engine_cap'].round()
	#
	# mean_values = df.groupby('type_of_vehicle')['engine_cap'].transform('mean')
	# df.loc[:, 'engine_cap'] = df['engine_cap'].fillna(mean_values)
	# df.loc[:, 'engine_cap'] = df['engine_cap'].round()
	#
	# # Step: fill in the missing values in column 'no_of_owners'
	# # We use the global average here
	# mean_values = df['no_of_owners'].mean()
	# df.loc[:, 'no_of_owners'] = df['no_of_owners'].fillna(mean_values)
	# df.loc[:, 'no_of_owners'] = df['no_of_owners'].round()
	#

	# Step: we remove rows where 'depreciation' or 'dereg_value' is null gere
	# Around 24,400 rows left after this step
	df = df.dropna(subset=['depreciation', 'dereg_value'])

	# # Step: we handle road_tax NaN values, around 2,600 in total.
	# # First, we fill up NaN values w.r.t. engine_cap
	# road_tax_dict = {}
	# for index, row in df.iterrows():
	# 	if row['engine_cap'] not in road_tax_dict.keys() or pd.isna(road_tax_dict[row['engine_cap']]):
	# 		road_tax_dict[row['engine_cap']] = row['road_tax']
	#
	# for index, row in df.iterrows():
	# 	if pd.isna(row['road_tax']):
	# 		df.loc[index, 'road_tax'] = road_tax_dict[row['engine_cap']]
	#
	# # After filling up with road_tax w.r.t. engine_cap, there are still around 1,000 NaN values
	# # We use linear approximation to fill up these values after using EDA
	# df_tmp = df.dropna(subset=['road_tax'])
	#
	# x = df_tmp['engine_cap'].values.reshape(-1, 1)
	# y = df_tmp['road_tax'].values
	# model = LinearRegression()
	# model.fit(x, y)
	#
	# # Find indices with missing road_tax values and predict
	# missing_indices = df[df['road_tax'].isnull()].index
	# x_missing = df.loc[missing_indices, 'engine_cap'].values.reshape(-1, 1)
	# y_pred = model.predict(x_missing)
	# df.loc[missing_indices, 'road_tax'] = [round(yi) for yi in y_pred]
	#
	# # Step: we handle missing values in column 'mileage' here
	# # We do random filling here
	# # missing_indices = df['mileage'].isnull()
	# # num_missing = missing_indices.sum()
	#
	# # random_values = np.random.randint(1000, 200001, size=num_missing)
	# # df.loc[missing_indices, 'mileage'] = random_values
	# mean_values = df.groupby('reg_year')['mileage'].transform('mean')
	# df.loc[:, 'mileage'] = df['mileage'].fillna(mean_values)
	# df.loc[:, 'mileage'] = df['mileage'].round()
	# # remaining NaN values apply the avg singapore mileage per year (17500)
	# df['mileage'].fillna((2024 - df['reg_year']) * 17500, inplace=True)
	#
	# ## --------------------------- omv --------------------
	# mean_values = df.groupby('model')['omv'].transform('mean')
	# df.loc[:, 'omv'] = df['omv'].fillna(mean_values)
	# df.loc[:, 'omv'] = df['omv'].round()
	#
	# mean_values = df.groupby('type_of_vehicle')['omv'].transform('mean')
	# df.loc[:, 'omv'] = df['omv'].fillna(mean_values)
	# df.loc[:, 'omv'] = df['omv'].round()
	#
	# ## --------------------------- curb_weight --------------------
	# mean_values = df.groupby('model')['curb_weight'].transform('mean')
	# df.loc[:, 'curb_weight'] = df['curb_weight'].fillna(mean_values)
	# df.loc[:, 'curb_weight'] = df['curb_weight'].round()
	#
	# mean_values = df.groupby('type_of_vehicle')['curb_weight'].transform('mean')
	# df.loc[:, 'curb_weight'] = df['curb_weight'].fillna(mean_values)
	# df.loc[:, 'curb_weight'] = df['curb_weight'].round()
	#
	# # Step: we handle missing values in column 'omv' and 'arf' here
	# # Since there are only around 100 missing values in total in this two columns
	# # We directly drop data points with these columns empty
	# df = df.dropna(subset=['omv', 'arf'])
	#
	# df = df.dropna(subset=['curb_weight'])

	return df


def HandlingMissingValueWithReference(df, df_test):
	model_dict_model = {}
	for index, row in df.iterrows():
		if row['model'] not in model_dict_model.keys() or pd.isna(model_dict_model[row['model']]):
			model_dict_model[row['model']] = row['make']

	for index, row in df_test.iterrows():
		if pd.isna(row['make']):
			if row['model'] in model_dict_model.keys():
				df_test.loc[index, 'make'] = model_dict_model[row['model']]

	return df_test


def HandlingMissingValueTest(df):
	# Step: Fill in 'power' group by
	mean_values = df.groupby('model')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	mean_values = df.groupby('make')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()

	mean_values = df.groupby('type_of_vehicle')['power'].transform('mean')
	df.loc[:, 'power'] = df['power'].fillna(mean_values)
	df.loc[:, 'power'] = df['power'].round()
	# # Step: remove the rows with the 'manufactured' column missing, 7 rows in total
	# # df = df.dropna(subset=['manufactured'])
	# df_test['manufactured'].fillna(df_test['reg_year'], inplace=True)
	#
	# # Step: fill in the missing values in column 'power'
	# # The power of cars for a certain model are very likely to be similar
	# # So we take the average values of power of each car model
	# # If there are still missing values, we take the average of 'type_of_vehicle'
	# mean_values = df.groupby('model')['power'].transform('mean')
	# df_test.loc[:, 'power'] = df_test['power'].fillna(mean_values).round()
	#
	# mean_values = df.groupby('type_of_vehicle')['power'].transform('mean')
	# df_test.loc[:, 'power'] = df_test['power'].fillna(mean_values).round()
	#
	# # Step: fill in the missing values in column 'engine_cap'
	# # We do the same as we did in step 3 here
	# mean_values = df.groupby('model')['engine_cap'].transform('mean')
	# df_test.loc[:, 'engine_cap'] = df_test['engine_cap'].fillna(mean_values).round()
	#
	# mean_values = df.groupby('type_of_vehicle')['engine_cap'].transform('mean')
	# df_test.loc[:, 'engine_cap'] = df_test['engine_cap'].fillna(mean_values).round()
	#
	# # Step: we handle missing values in column 'mileage' here
	# mean_values = df.groupby('reg_year')['mileage'].transform('mean')
	# df_test.loc[:, 'mileage'] = df_test['mileage'].fillna(mean_values).round()
	# # remaining NaN values apply the avg singapore mileage per year (17500)
	# df_test['mileage'].fillna((2024 - df_test['reg_year']) * 17500, inplace=True)
	#
	# ## --------------------------- omv --------------------
	# mean_values = df.groupby('model')['omv'].transform('mean')
	# df_test.loc[:, 'omv'] = df_test['omv'].fillna(mean_values).round()
	#
	# mean_values = df.groupby('type_of_vehicle')['omv'].transform('mean')
	# df_test.loc[:, 'omv'] = df_test['omv'].fillna(mean_values).round()
	#
	# ## --------------------------- curb_weight --------------------
	# mean_values = df.groupby('model')['curb_weight'].transform('mean')
	# df_test.loc[:, 'curb_weight'] = df_test['curb_weight'].fillna(mean_values).round()
	#
	# mean_values = df.groupby('type_of_vehicle')['curb_weight'].transform('mean')
	# df_test.loc[:, 'curb_weight'] = df_test['curb_weight'].fillna(mean_values).round()

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


def HandlingMissingValueWithImpute(df, columns):
	imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)

	df_imputed_values = imputer.fit_transform(df[columns])
	df_imputed = pd.DataFrame(df_imputed_values, columns=columns, index=df.index)

	df_result = df.copy()
	df_result[columns] = df_imputed
	total_nulls = df_result.isnull().sum().sum()

	print("NaN values after handling: ", total_nulls)

	return df_result


def HandlingMissingValueWithImputeReference(df_target, df_reference, columns):
	combined_df = pd.concat([df_target[columns], df_reference[columns]], ignore_index=True)
	imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
	imputed_array = imputer.fit_transform(combined_df)
	imputed_target = imputed_array[:len(df_target)]

	df_imputed = pd.DataFrame(imputed_target, columns=columns, index=df_target.index)

	df_result = df_target.copy()
	df_result[columns] = df_imputed

	total_nulls = df_result.isnull().sum().sum()

	print(df_result.head())
	print("NaN values after handling: ", total_nulls)

	return df_result


# We apply data encoding here
def DataEncoding(df):
	# We handle the attribute 'category' here
	# df = HandlingMissingValue(df)
	return df


# Remove outlier by group
def OutlierRemoval(df, group_column, target_column):
	# For column omv, we apply 3-sigma law to remove outliers by group
	df = remove_outliers_by_group(df, group_column, target_column)

	num_records, num_attributes = df.shape
	print("There are {} data points, each with {} attributes".format(num_records, num_attributes))
	return df


def DataAugmentation(df):
	model_counts = df['model'].value_counts()
	models_to_augment = model_counts[model_counts < 20].index
	augmented_data = pd.DataFrame()

	for model in models_to_augment:
		subset = df[df['model'] == model]
		num_copies = 5
		if num_copies > 0:
			augmented_subset = pd.concat([subset] * (num_copies + 1), ignore_index=True)
		else:
			augmented_subset = subset

		augmented_data = pd.concat([augmented_data, augmented_subset], ignore_index=True)

	df_augmented = pd.concat([df, augmented_data], ignore_index=True)
	return df_augmented


def DeleteColumns(df):
	return df.drop(columns=columns_to_delete)
