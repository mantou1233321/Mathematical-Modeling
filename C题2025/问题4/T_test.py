import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 读数据
df = pd.read_csv('附件_wash_nv.csv')   # ← 换成你的文件

# 2. 指定要在 t 检验和图中用的列
VALUE_COL = '13号染色体的Z值'   # ← 想检验的指标
GROUP_COL = '染色体的非整倍体'   # ← 0/1 分组列

# 3. 独立样本 t 检验
group0 = df[df[GROUP_COL] == 0][VALUE_COL]
group1 = df[df[GROUP_COL] == 1][VALUE_COL]
t, p = ttest_ind(group0, group1, equal_var=False)   # Welch’s t
print(f'Welch t={t:.3f}, p={p:.4g}')

# 4. 散点图：x 轴随意（这里用索引），y 轴为指标
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x=df.index, y=VALUE_COL,
                hue=GROUP_COL, palette={0: 'tab:blue', 1: 'red'},
                s=40, legend='full')

plt.title(f'{VALUE_COL} 非整倍体 vs 对照\nWelch t-test  p = {p:.4g}')
plt.xlabel('样本序号')
plt.ylabel(VALUE_COL)
plt.legend(title=GROUP_COL, loc='best')
plt.tight_layout()
plt.show()