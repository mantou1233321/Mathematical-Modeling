import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 读数据 -------------------------------------------------------------
df = pd.read_csv('C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_3.csv').dropna()

y = df['Y染色体浓度']
X = df[['检测抽血次数', '孕妇BMI', '原始读段数', 'GC含量', 'X染色体浓度', '检测孕周']]
X = sm.add_constant(X)  # 截距

# 2. 多元线性回归（稳健标准误）------------------------------------------
model = sm.OLS(y, X).fit(cov_type='HC3')
print(model.summary())  # 显著性表

# 3. 提取显著性结果 -----------------------------------------------------
pvals = model.pvalues
sig = pvals[pvals < 0.05]
print('\n在 α=0.05 下显著变量：')
print(sig)

# 4. 可视化（单独绘制每张图） -------------------------------------------
for col in X.columns[1:]:  # 遍历所有自变量
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col], y, alpha=0.6, label='数据点')
    sns.regplot(x=col, y='Y染色体浓度', data=df, scatter=False, color='red', line_kws={'lw': 2}, label='回归线')
    plt.title(f'{col} 与 Y染色体浓度的关系 (p={pvals[col]:.3f})')
    plt.xlabel(col)
    plt.ylabel('Y染色体浓度')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{col}_vs_Y染色体浓度.png', dpi=300)  # 保存为单独的图片
    plt.show()

# 5. 残差 vs 拟合图（模型诊断） ------------------------------------------
plt.figure(figsize=(8, 6))
yhat = model.fittedvalues
resid = model.resid
plt.scatter(yhat, resid, alpha=0.6)
plt.axhline(0, color='red', ls='--')
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.title('残差 vs 拟合值')
plt.tight_layout()
plt.savefig('残差_vs_拟合值.png', dpi=300)  # 保存残差图
plt.show()