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
df = pd.read_csv('附件_wash_only3.csv')

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

# 数据预处理：日期转换
date_cols = ['末次月经', '检测日期']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        # 将日期转换为距离最早日期的天数
        min_date = df[col].min()
        df[f'{col}_数值'] = (df[col] - min_date).dt.days
        # 替换变量列表中的日期列为数值列
        variables[variables.index(col)] = f'{col}_数值'

# 提取分析所需数据
analysis_cols = ['Y染色体浓度'] + variables
analysis_df = df[analysis_cols].dropna()  # 去除分析列中的缺失值

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

pearson_df = pd.DataFrame(pearson_results).T
spearman_df = pd.DataFrame(spearman_results).T
print("皮尔逊相关性分析结果：")
print(pearson_df)
print("\n斯皮尔曼相关性分析结果：")
print(spearman_df)

# 可视化相关性矩阵
# 选取相关性分析的数值列
corr_matrix_pearson = analysis_df.corr(method='pearson')
corr_matrix_spearman = analysis_df.corr(method='spearman')

# Y染色体浓度与其他变量的相关性
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

# 似然比检验
print("\n似然比检验：")

# 定义嵌套模型（假设去掉部分变量）
reduced_variables = variables[:8]  # 仅保留前8个变量作为嵌套模型
X_reduced = analysis_df[reduced_variables]

# 拟合嵌套模型
reg_reduced = LinearRegression()
reg_reduced.fit(X_reduced, y)

# 计算完整模型和嵌套模型的残差平方和（RSS）
RSS_full = np.sum((y - reg.predict(X)) ** 2)
RSS_reduced = np.sum((y - reg_reduced.predict(X_reduced)) ** 2)

# 计算似然比统计量
n = len(y)  # 样本数
p_full = X.shape[1]  # 完整模型的参数个数
p_reduced = X_reduced.shape[1]  # 嵌套模型的参数个数
df_diff = p_full - p_reduced  # 自由度差

LRT_stat = (RSS_reduced - RSS_full) / (RSS_full / (n - p_full))
p_value_LRT = 1 - stats.f.cdf(LRT_stat, df_diff, n - p_full)

# 输出结果
print(f"完整模型 RSS = {RSS_full:.4f}")
print(f"嵌套模型 RSS = {RSS_reduced:.4f}")
print(f"似然比统计量 = {LRT_stat:.4f}")
print(f"自由度差 = {df_diff}")
print(f"p值 = {p_value_LRT:.4f}")

if p_value_LRT < 0.05:
    print("模型改进显著 (p < 0.05)")
else:
    print("模型改进不显著 (p >= 0.05)")

print("分析完成，结果已保存为Excel和图片文件")