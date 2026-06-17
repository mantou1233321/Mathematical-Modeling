import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('C:/Users/26218/Desktop/2025/2025题目/C题/附件（女）.csv')

# 1. 将检测孕周列转换为字符串类型后，再转换为纯数值
# 使用原始字符串处理正则表达式，避免转义警告
df['检测孕周'] = df['检测孕周'].astype(str).str.extract(r'(\d+)').astype(float)

# 2. 设定阈值进行筛选
# df = df[ (df['在参考基因组上比对的比例']>=0.75) & (df['GC含量']>=0.35) & (df['GC含量']<=0.6)]

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
                    '重复读段的比例', '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                    '21号染色体的Z值', 'X染色体的Z值',  'X染色体浓度',
                    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
                    '怀孕次数', '生产次数']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# 7. 删除含有缺失数据的记录
df = df.dropna()

# 8. 鲁棒地标准化日期列
date_cols = ['末次月经', '检测日期']
for col in date_cols:
    if col not in df.columns:
        continue
    # 去空格、去空串、去纯标点
    s = df[col].astype(str).str.strip().replace(r'^[,;:\s]*$', '', regex=True)
    # 无法解析的→NaT，再统一格式化
    s = pd.to_datetime(s, errors='coerce', dayfirst=False)
    df[col] = s.dt.strftime('%Y-%m-%d')          # NaT 会变成 NaN


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
df = df.dropna()
# 将结果保存为 csv 文件
csv_path = 'C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_nv.csv'
print("done!")
df.to_csv(csv_path, index=False)