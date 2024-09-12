import pandas as pd


def ReadCSV(file_path):
	with open(file_path, 'r') as file:
    	df = pd.read_csv(file)

	value_counts = df['model'].value_counts()

    print(value_counts)
    # 步骤2: 筛选出现次数为1的值
    single_occurrences = value_counts[value_counts == 1]

    # 步骤3: 筛选原始数据中这些值的行
    filtered_df = df[df['Column'].isin(single_occurrences.index)]

    # 步骤4: 计算数量
    number_of_single_occurrence_rows = filtered_df.shape[0]

    # 打印结果
    print("Number of rows where the value in 'Column' occurs only once:", number_of_single_occurrence_rows)