import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We apply data cleaning here
# This may include data reduction, data transformation, data filing, etc.
# We use the function CleanData() to do the cleaning
# The cleaning steps are specified in the CleanData() function
columns_to_delete = [
	'listing_id',
	'title',
	'description',
	'original_reg_date',
	'fuel_type',  # We may consider filling up this column later
	'original_reg_date',
	'opc_scheme',
	'lifespan',
	'eco_category',
	'indicative_price',
	'road_tax',
	'mileage',
	'omv',
	'arf',
	'features',
	'accessories'
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

	# Removing useless columns
	df = DeleteColumns(df)
	total_nan_count = df.isna().sum().sum()
	print("Missing values after cleaning:", total_nan_count)

	# Data encoding
	df = DataEncoding(df)

	# Data
	WriteCSV(df, out_path)


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

	# Then concatenate column 'model' and 'make' to avoid duplicate 'model' values
	df.loc[:, 'model'] = df['make'].str.cat(df['model'], sep='-')

	# Step: remove the rows with the 'manufactured' column missing, 7 rows in total
	df = df.dropna(subset=['manufactured'])

	# Step: fill in the missing values in column 'curb_weight'
	# The curb weight of cars for a certain model are very likely to be similar
	# So we take the average values
	# If there are still missing values, we take the average of 'type_of_vehicle'
	mean_values = df.groupby('model')['curb_weight'].transform('mean')
	df.loc[:, 'curb_weight'] = df['curb_weight'].fillna(mean_values)
	df.loc[:, 'curb_weight'] = df['curb_weight'].round()

	mean_values = df.groupby('type_of_vehicle')['curb_weight'].transform('mean')
	df.loc[:, 'curb_weight'] = df['curb_weight'].fillna(mean_values)
	df.loc[:, 'curb_weight'] = df['curb_weight'].round()

	# Step: fill in the missing values in column 'power'
	# We do the same as we did in step 3 here
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
	# Around 24400 rows left after this step
	df = df.dropna(subset=['depreciation', 'dereg_value'])
	return df


def DataEncoding(df):
	df.loc[:, 'transmission'] = df['transmission'].map({'auto': 0, 'manual': 1})

	return df


def DeleteColumns(df):
	return df.drop(columns=columns_to_delete)
