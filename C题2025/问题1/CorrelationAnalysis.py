import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.inspection import permutation_importance


# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载清洗后的数据
df = pd.read_csv('C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3.csv')



# 定义需要分析的变量列表
variables = [
    
    '年龄', 
    '末次月经', 'IVF妊娠', '检测日期', '检测抽血次数', '检测孕周', 
    '孕妇BMI', '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', 
    '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值', 
    '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'X染色体浓度', 
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
    '被过滤掉读段数的比例', '染色体的非整倍体', '怀孕次数', '生产次数'
]

# 数据预处理：将日期转换为数值（天数）以便计算相关性
date_cols = ['末次月经', '检测日期']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        # 将日期转换为距离最早日期的天数
        min_date = df[col].min()
        df[f'{col}_数值'] = (df[col] - min_date).dt.days
        # 替换变量列表中的日期列为数值列
        variables[variables.index(col)] = f'{col}_数值'

# 确保Y染色体浓度列存在
if 'Y染色体浓度' not in df.columns:
    raise ValueError("数据中未找到'Y染色体浓度'列，请检查列名是否正确")

# 提取分析所需数据
analysis_cols = ['Y染色体浓度'] + variables
analysis_df = df[analysis_cols].dropna()  # 去除分析列中的缺失值

# # 对非分类变量进行离散化
# non_categorical_vars = []
# for var in variables:
#     # 判断条件：唯一值数量>2（排除二值分类变量）且为数值型
#     if analysis_df[var].nunique() > 2 and pd.api.types.is_numeric_dtype(analysis_df[var]):
#         non_categorical_vars.append(var)

# num_bins = 5  # 分箱数量，可根据数据分布调整
# for var in non_categorical_vars:
#     try:
#         # 使用qcut进行等频分箱，添加duplicates参数处理重复边界
#         analysis_df[var] = pd.qcut(
#             analysis_df[var], 
#             q=num_bins, 
#             labels=False,
#             duplicates='drop'  # 关键修复：删除重复的分箱边界
#         )
#         # 保存分箱区间（单独存放在另一个DataFrame中，不参与建模）
#         # 新增：创建分箱区间的独立存储，不影响主数据
#         bin_intervals = pd.qcut(
#             analysis_df[var], 
#             q=num_bins, 
#             duplicates='drop'
#         )
#         # 打印分箱区间信息（可选）
#         print(f"{var} 分箱区间: {bin_intervals.cat.categories.tolist()}")
#     except Exception as e:
#         print(f"变量 {var} 分箱时出错: {str(e)}，将使用原始值")

# # 关键修复：确保仅保留数值型变量用于相关性分析
# # 筛选出所有数值型列（排除可能存在的非数值类型）
# numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
# # 确保Y染色体浓度和自变量都在数值列中
# analysis_numeric_df = analysis_df[numeric_cols]


# 1. 皮尔逊相关性分析（适用于线性关系）
pearson_results = {}
for var in variables:
    corr, p_value = stats.pearsonr(analysis_df['Y染色体浓度'], analysis_df[var])
    pearson_results[var] = {
        '相关系数': round(corr, 4),
        'P值': round(p_value, 4),
        '显著性': '显著' if p_value < 0.05 else '不显著'
    }

# 2. 斯皮尔曼相关性分析（适用于非线性/序数关系）
spearman_results = {}
for var in variables:
    corr, p_value = stats.spearmanr(analysis_df['Y染色体浓度'], analysis_df[var])
    spearman_results[var] = {
        '相关系数': round(corr, 4),
        'P值': round(p_value, 4),
        '显著性': '显著' if p_value < 0.05 else '不显著'
    }

# 将结果转换为DataFrame
pearson_df = pd.DataFrame(pearson_results).T
spearman_df = pd.DataFrame(spearman_results).T

# 打印结果
print("皮尔逊相关性分析结果：")
print(pearson_df)
print("\n斯皮尔曼相关性分析结果：")
print(spearman_df)

# 可视化相关性矩阵
# 选取相关性分析的数值列
corr_matrix_pearson = analysis_df.corr(method='pearson')
corr_matrix_spearman = analysis_df.corr(method='spearman')

# 绘制皮尔逊相关性热图
plt.figure(figsize=(18, 12))
sns.heatmap(
    corr_matrix_pearson, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    fmt='.2f', 
    linewidths=0.5
)
plt.title('皮尔逊相关性矩阵热图（含年龄和计算的BMI）')
plt.tight_layout()
plt.savefig('皮尔逊相关性热图_含年龄和BMI.png', dpi=300)
plt.show()

# 绘制斯皮尔曼相关性热图
plt.figure(figsize=(18, 12))
sns.heatmap(
    corr_matrix_spearman, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    fmt='.2f', 
    linewidths=0.5
)
plt.title('斯皮尔曼相关性矩阵热图（含年龄和计算的BMI）')
plt.tight_layout()
plt.savefig('斯皮尔曼相关性热图_含年龄和BMI.png', dpi=300)
plt.show()

# 单独展示Y染色体浓度与其他变量的相关性
plt.figure(figsize=(14, 10))
y_corr_pearson = corr_matrix_pearson['Y染色体浓度'].sort_values(ascending=False)
y_corr_pearson.drop('Y染色体浓度').plot(kind='bar')
plt.title('Y染色体浓度与各变量的皮尔逊相关系数（含年龄和计算的BMI）')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.3, label='弱相关阈值')
plt.axhline(y=-0.3, color='r', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Y染色体浓度_皮尔逊相关性_含年龄和BMI.png', dpi=300)
plt.show()

plt.figure(figsize=(14, 10))
y_corr_spearman = corr_matrix_spearman['Y染色体浓度'].sort_values(ascending=False)
y_corr_spearman.drop('Y染色体浓度').plot(kind='bar')
plt.title('Y染色体浓度与各变量的斯皮尔曼相关系数（含年龄和计算的BMI）')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.3, label='弱相关阈值')
plt.axhline(y=-0.3, color='r', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Y染色体浓度_斯皮尔曼相关性_含年龄和BMI.png', dpi=300)
plt.show()

# 多元线性回归分析
# 以Y染色体浓度为因变量，其余变量为自变量
X = analysis_df[variables]
y = analysis_df['Y染色体浓度']

# 建立回归模型
reg = LinearRegression()
reg.fit(X, y)

# 输出回归系数
coef_df = pd.DataFrame({
    '变量': variables,
    '回归系数': reg.coef_
})
print("\n多元线性回归各变量系数：")
print(coef_df)

# 使用RANSAC鲁棒回归
ransac = RANSACRegressor(estimator=LinearRegression(), random_state=0)
ransac.fit(X, y)
ransac_coef_df = pd.DataFrame({
    '变量': variables,
    'RANSAC回归系数': ransac.estimator_.coef_
})
print("\nRANSAC鲁棒回归各变量系数：")
print(ransac_coef_df)

# 置换特征重要性分析
perm_importance = permutation_importance(reg, X, y, n_repeats=30, random_state=0)
perm_df = pd.DataFrame({
    '变量': variables,
    '置换特征重要性均值': perm_importance.importances_mean,
    '置换特征重要性标准差': perm_importance.importances_std
}).sort_values(by='置换特征重要性均值', ascending=False)

print("\n置换特征重要性：")
print(perm_df)

# 保存结果为Excel
with pd.ExcelWriter('Y染色体浓度相关性分析结果_含年龄和BMI.xlsx') as writer:
    pearson_df.to_excel(writer, sheet_name='皮尔逊相关性')
    spearman_df.to_excel(writer, sheet_name='斯皮尔曼相关性')
    coef_df.to_excel(writer, sheet_name='多元线性回归系数', index=False)
    ransac_coef_df.to_excel(writer, sheet_name='RANSAC回归系数', index=False)
    perm_df.to_excel(writer, sheet_name='置换特征重要性', index=False)


print("分析完成，结果已保存为Excel和图片文件")