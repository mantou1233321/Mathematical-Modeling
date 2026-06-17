# 9. 使用3σ法则去除异常值
# # 只对数值型列应用3σ法则
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# # 初始化一个全为True的掩码，表示所有行都保留
# mask = pd.Series([True] * len(df), index=df.index)

# for col in numeric_columns:
#     # 计算均值和标准差
#     mean = df[col].mean()
#     std = df[col].std()
    
#     # 计算3σ上下限
#     lower_bound = mean - 3 * std
#     upper_bound = mean + 3 * std
    
#     # 更新掩码，保留在3σ范围内的数据
#     mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

# # 应用掩码过滤异常值
# df = df[mask]