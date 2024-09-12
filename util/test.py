import pandas as pd

# 创建一个示例 DataFrame
data = {'Column': ['A', 'B', 'A', 'C', 'B', 'D', 'E', 'F', 'G', 'H']}
df = pd.DataFrame(data)

# 步骤1: 计算每个值的出现次数
value_counts = df['Column'].value_counts()

# 步骤2: 筛选出现次数为1的值
single_occurrences = value_counts[value_counts == 1]

# 步骤3: 筛选原始数据中这些值的行
filtered_df = df[df['Column'].isin(single_occurrences.index)]

# 步骤4: 计算数量
number_of_single_occurrence_rows = filtered_df.shape[0]

# 打印结果
print("Number of rows where the value in 'Column' occurs only once:", number_of_single_occurrence_rows)
