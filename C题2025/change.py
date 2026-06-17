import pandas as pd

# 设置文件路径（直接在这里修改文件名）
input_file = "C:/Users/26218/Desktop/2025/2025题目/C题/附件.xlsx" # 输入的Excel文件名
output_file = "C:/Users/26218/Desktop/2025/2025题目/C题/附件（女）.csv" # 输出的CSV文件名

# 读取Excel文件
df = pd.read_excel(input_file,sheet_name="女胎检测数据")

# 保存为CSV文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f'转换完成: {input_file} -> {output_file}')
print(f'数据形状: {df.shape}行 x {df.shape[1]}列')
print('列名:', list(df.columns))