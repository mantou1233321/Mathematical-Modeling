import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('附件.csv')

# 将检测孕周转换为纯数值
df['检测孕周'] = df['检测孕周'].astype(str).str.extract(r'(\d+)').astype(float)

# 设定阈值进行筛选，数据处理
df = df[(df['在参考基因组上比对的比例'] >= 0.75) & 
        (df['GC含量'] >= 0.35) & 
        (df['GC含量'] <= 0.6)]
df['染色体的非整倍体'] = df['染色体的非整倍体'].notnull().astype(int)
df['怀孕次数'] = df['怀孕次数'].str.replace('≥3', '3').astype(int)
df['怀孕次数'] = df['怀孕次数'].apply(lambda x: 3 if x >= 3 else x)
df['IVF妊娠'] = df['IVF妊娠'].map({'自然受孕': 0, 'IUI（人工授精）': 1})
df['胎儿是否健康'] = df['胎儿是否健康'].map({'是': 1, '否': 0})

# 删除BMI极端数据
print(f"原始数据条数: {len(df)}")
df = df[(df['孕妇BMI'] >= 27) & (df['孕妇BMI'] <= 44)]
print(f"筛选后数据条数 (27 ≤ BMI ≤ 44): {len(df)}")

# 删除含有缺失数据的记录
df = df.dropna()

# 标准化日期格式
if '末次月经' in df.columns:
    df['末次月经'] = pd.to_datetime(df['末次月经'], format='mixed').dt.strftime('%Y-%m-%d')
if '检测日期' in df.columns:
    df['检测日期'] = pd.to_datetime(df['检测日期'], format='mixed').dt.strftime('%Y-%m-%d')

# 对孕妇BMI列进行K-means聚类
bmi_data = df['孕妇BMI'].values.reshape(-1, 1)

# 数据标准化
scaler = StandardScaler()
bmi_scaled = scaler.fit_transform(bmi_data)

# 使用肘部法则确定最佳聚类数量
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bmi_scaled)
    wcss.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Square)')
plt.title('肘部法则 - 确定最佳聚类数量')
plt.grid(True)
plt.show()

# 根据肘部法则选择最佳k值（通常选择拐点处的k值）
# 这里假设选择4个聚类（低、中、高BMI）
optimal_k = 4
# 使用K-means进行聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['BMI_cluster'] = kmeans.fit_predict(bmi_scaled)
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

print("聚类中心（原始BMI值）:")
for i, center in enumerate(cluster_centers_original):
    print(f"聚类 {i}: {center[0]:.2f}")

# 分析每个聚类的统计信息
print("\n各聚类统计信息:")
cluster_stats = df.groupby('BMI_cluster')['孕妇BMI'].agg(['count', 'mean', 'std', 'min', 'max'])
print(cluster_stats)

# 可视化聚类结果
plt.figure(figsize=(12, 6))

# 子图1: 聚类分布直方图
plt.subplot(1, 2, 1)
for cluster in range(optimal_k):
    cluster_data = df[df['BMI_cluster'] == cluster]['孕妇BMI']
    plt.hist(cluster_data, alpha=0.7, label=f'聚类{cluster}', bins=20)
plt.xlabel('孕妇BMI')
plt.ylabel('频数')
plt.title('BMI聚类分布 (27 ≤ BMI ≤ 44)')
plt.legend()
plt.grid(True)

# 子图2: 箱线图
plt.subplot(1, 2, 2)
sns.boxplot(x='BMI_cluster', y='孕妇BMI', data=df)
plt.title('各聚类BMI分布箱线图')
plt.grid(True)

plt.tight_layout()
plt.show()

# 分析聚类与胎儿健康的关系
if '胎儿是否健康' in df.columns:
    print("\nBMI聚类与胎儿健康的关系:")
    health_by_cluster = pd.crosstab(df['BMI_cluster'], df['胎儿是否健康'], 
                                   normalize='index') * 100
    print(health_by_cluster)
    
    # 可视化
    plt.figure(figsize=(8, 6))
    health_by_cluster.plot(kind='bar', stacked=True)
    plt.title('各BMI聚类中胎儿健康状况分布')
    plt.xlabel('BMI聚类')
    plt.ylabel('百分比 (%)')
    plt.legend(['不健康', '健康'])
    plt.grid(True)
    plt.show()

# 保存带有聚类标签的数据
df.to_csv('处理后的数据_带聚类.csv', index=False, encoding='utf-8-sig')
print("BMI聚类完成，数据已保存！")