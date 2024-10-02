import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# listing_id            	Identical ID
# title						Indicating cars' titles, very sparse with 7000 types in 25000 data points
# make                   95个类，可以编码使用
# model						799个类  可以编码使用  可以考虑把少的集中起来定义为其他。例如少于10的类别
# description			删了 太杂
# manufactured				71个类，可以编码使用
# original_reg_date			删了
# reg_date					可以尝试编码日期只保留年
# type_of_vehicle			良好分类  可以编码
# category					两百多种，可以考虑把少的集中起来定义为其他。例如少于10的类别 编码
# transmission				良好分类  编码
# curb_weight				可以编码变成区间，也可以不动
# power						空值比较多 可以编码变成区间，也可以不动
# fuel_type						删
# engine_cap                建议不动，和road_tax有关系
# no_of_owners					用原值 不做处理

# depreciation                  有空值但比较少直接去掉空行
# coe                       没有空值    这个值怎么搞还在考虑
# road_tax                      和 engine_cap 有对应关系  根据engine填充 建议不动
# dereg_value               有空值但比较少直接去掉空行 这个值怎么搞还在考虑
# mileage                  比较集中  填充个人用的随机值

# omv						和下面的arf似乎有对应关系  空行直接删了 可以编码变成区间，也可以不动
# arf                       空行直接删了 可以编码变成区间，也可以不动
# opc_scheme			删
# lifespan				删
# eco_category				删
# features					 删 和下面的accessory感觉差不多  文字属性 暂时先删掉  不过提示属于worth属性，感觉可以挖掘一下
# accessories               删
# indicative_price			删
# price

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

	# Removing useless columns
	df = DeleteColumns(df)
	total_nan_count = df.isna().sum().sum()
	print("Missing values after cleaning:", total_nan_count)

	# Data encoding
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
	#
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


# 可以编码的列：make,model,reg_date,type_of_vehicle,category,transmission
def DataEncoding(df):
	# 使用 pandas 的 factorize 方法编码 'make' 列
	df['make_encoded'], unique_values = pd.factorize(df['make'])

	model_counts = df['model'].value_counts()
	to_other = model_counts[model_counts < 10].index
	# 将这些类别替换为“其他”
	df['model'] = df['model'].apply(lambda x: 'Other' if x in to_other else x)
	df['model_encoded'], unique_values = pd.factorize(df['model'])

	df['manufactured_encoded'], unique_values = pd.factorize(df['manufactured'])

	# 将 'reg_date' 列转换为 '月-年' 格式  然后编码
	df['reg_date'] = pd.to_datetime(df['reg_date'], format='%d-%b-%Y')
	df['reg_date'] = df['reg_date'].dt.strftime('%Y')
	df['reg_date_encoded'], unique_values = pd.factorize(df['reg_date'])

	df['type_of_vehicle_encoded'], unique_values = pd.factorize(df['type_of_vehicle'])

	category_counts = df['category'].value_counts()
	to_other = category_counts[category_counts < 10].index
	# 将这些类别替换为“其他”
	df['category'] = df['category'].apply(lambda x: 'Other' if x in to_other else x)
	df['category_encoded'], unique_values = pd.factorize(df['category'])

	df['transmission'], unique_values = pd.factorize(df['transmission'])
	# df.loc[:, 'transmission'] = df['transmission'].map({'auto': 0, 'manual': 1})

	return df


def DeleteColumns(df):
	return df.drop(columns=columns_to_delete)
