import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('C:/Users/26218/Desktop/2025/2025题目/C题/附件.csv')

# 1. 将检测孕周列转换为字符串类型后，再转换为纯数值
# 使用原始字符串处理正则表达式，避免转义警告
df['检测孕周'] = df['检测孕周'].astype(str).str.extract(r'(\d+)').astype(float)



# 3. 处理染色体的非整倍体列
df['染色体的非整倍体'] = df['染色体的非整倍体'].notnull().astype(int)

# 4. 处理怀孕次数列，先将类似 '≥3' 的值替换为 3
df['怀孕次数'] = df['怀孕次数'].str.replace('≥3', '3').astype(int)
df['怀孕次数'] = df['怀孕次数'].apply(lambda x: 3 if x >= 3 else x)

# 5. 对 IVF 妊娠和胎儿是否健康列进行数学离散化
df['IVF妊娠'] = df['IVF妊娠'].map({'自然受孕': 0, 'IUI（人工授精）': 1})
df['胎儿是否健康'] = df['胎儿是否健康'].map({'是': 1, '否': 0})

# 6. 数据标准化
scaler = StandardScaler()
columns_to_scale = ['检测孕周', '年龄', '身高', '体重', '孕妇BMI', '原始读段数', '在参考基因组上比对的比例',
                    '重复读段的比例', '唯一比对的读段数  ', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                    '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度', 'X染色体浓度',
                    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
                    '怀孕次数', '生产次数']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# 7. 删除含有缺失数据的记录
df = df.dropna()

# 8. 标准化日期格式为YYYY-MM-DD，使用mixed格式处理不同日期格式
if '末次月经' in df.columns:
    # 处理混合格式的日期
    df['末次月经'] = pd.to_datetime(df['末次月经'], format='mixed').dt.strftime('%Y-%m-%d')
if '检测日期' in df.columns:
    # 处理可能的数字日期格式（如20230429）
    df['检测日期'] = pd.to_datetime(df['检测日期'], format='mixed').dt.strftime('%Y-%m-%d')

# 9. 使用3σ法则去除异常值
# 只对数值型列应用3σ法则
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# 初始化一个全为True的掩码，表示所有行都保留
mask = pd.Series([True] * len(df), index=df.index)

for col in numeric_columns:
    # 计算均值和标准差
    mean = df[col].mean()
    std = df[col].std()
    
    # 计算3σ上下限
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    # 更新掩码，保留在3σ范围内的数据
    mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

# 应用掩码过滤异常值
df = df[mask]

# 将结果保存为 csv 文件
csv_path = 'C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3xita.csv'
print("done!")
df.to_csv(csv_path, index=False)