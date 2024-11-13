import pandas as pd
import numpy as np

# 读取 CSV 文件
df1 = pd.read_csv('./data/xgb_result.csv')
df1 = pd.read_csv('./data/xgb_by_model_result.csv')
df2 = pd.read_csv('./data/submission_006.csv')

# 确保 'Predicted' 列存在并且行数相同
if 'Predicted' in df1.columns and 'Predicted' in df2.columns:
    if len(df1) == len(df2):
        # 计算 RMSE
        rmse = np.sqrt(((df1['Predicted'] - df2['Predicted']) ** 2).mean())
        print(f"The RMSE between the two files is: {rmse}")

        # 计算每行的绝对误差
        df1['Error'] = np.abs(df1['Predicted'] - df2['Predicted'])
        df1['GT'] = df2['Predicted']

        # 找出差距特别大的行，假设阈值为 10，可以根据需要调整
        threshold = 1000
        large_errors = df1[df1['Error'] > threshold]

        # 打印差距特别大的行
        if not large_errors.empty:
            print("Rows with large differences:")
            print(large_errors)

        else:
            print("No rows with differences larger than the threshold.")

    else:
        print("The two files have different numbers of rows.")
else:
    print("The column 'Predicted' does not exist in one or both files.")

large_errors.to_csv('./data/error.csv', index=False)
print(len(large_errors))