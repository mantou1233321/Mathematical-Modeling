import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTOptimizer:
    def visualize_results(self):
        """
        可视化每组的最优检测时点和风险分量
        """
        print("\n=== 可视化每组的最优检测时点和风险分量 ===")

        for cluster, results in self.optimal_times.items():
            delay_risk = results['delay_risk']
            failure_risk = results['failure_risk']
            optimal_time = results['optimal_time']

            # 绘制风险分量柱状图
            plt.figure(figsize=(8, 6))
            plt.bar(['延迟风险', '失败风险'], [delay_risk, failure_risk], color=['blue', 'orange'])
            plt.title(f"BMI分组{cluster} - 最优检测时点: {optimal_time:.2f}周")
            plt.ylabel("风险值")
            plt.grid(axis='y')
            plt.show()
    
    
    def analyze_error_impact(self, num_simulations=1000):
        """
        分析误差对最优检测时点的影响，并绘制风险曲线
        Parameters:
        num_simulations: int, 模拟次数（默认1000）
        """
        print("\n=== 分析误差对最优检测时点的影响 ===")

        for cluster, model_params in self.regression_models.items():
            mean_bmi = self.bmi_groups[cluster]['mean_bmi']
            sigma = model_params['sigma']

            optimal_times = []

            for _ in range(num_simulations):
                # 添加随机误差到模型参数
                perturbed_params = model_params['params'] + np.random.normal(0, sigma, size=len(model_params['params']))

                # 定义目标函数
                def objective(t):
                    predicted_concentration = (
                        perturbed_params[0] +
                        perturbed_params[1] * mean_bmi +
                        perturbed_params[2] * np.log(t + 1) +
                        perturbed_params[3] * np.sqrt(mean_bmi)
                    )
                    z_score = (0.04 - predicted_concentration) / sigma
                    failure_risk = stats.norm.cdf(z_score)
                    delay_risk = self.delay_risk(t)
                    return self.lambda_weight * delay_risk + (1 - self.lambda_weight) * failure_risk

                # 优化寻找最优检测时点
                result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
                optimal_times.append(result.x)

            # 计算误差影响的统计量
            mean_time = np.mean(optimal_times)
            std_time = np.std(optimal_times)

            # 计算10%容忍范围内的比例
            lower_bound = mean_time * 0.9
            upper_bound = mean_time * 1.1
            within_tolerance = np.sum((np.array(optimal_times) >= lower_bound) & (np.array(optimal_times) <= upper_bound)) / num_simulations

            print(f"\nBMI分组{cluster}:")
            print(f"  最优检测时点均值: {mean_time:.2f}周")
            print(f"  最优检测时点标准差: {std_time:.2f}周")
            print(f"  10%容忍范围内比例: {within_tolerance:.2%}")

            # 绘制误差影响的分布图
            plt.figure(figsize=(8, 6))
            plt.hist(optimal_times, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(mean_time, color='red', linestyle='--', label=f'均值: {mean_time:.2f}周')
            plt.axvline(lower_bound, color='green', linestyle='--', label=f'10%下界: {lower_bound:.2f}周')
            plt.axvline(upper_bound, color='orange', linestyle='--', label=f'10%上界: {upper_bound:.2f}周')
            plt.title(f"BMI分组{cluster} - 最优检测时点分布")
            plt.xlabel("最优检测时点 (周)")
            plt.ylabel("频数")
            plt.legend()
            plt.grid()
            plt.show()

            # 绘制风险曲线
            t_values = np.linspace(10, 25, 100)
            delay_risks = [self.delay_risk(t) for t in t_values]
            failure_risks = [self.failure_risk(t, mean_bmi, model_params) for t in t_values]
            total_risks = [self.total_risk(t, mean_bmi, model_params) for t in t_values]

            plt.figure(figsize=(10, 6))
            plt.plot(t_values, delay_risks, label='延迟风险', color='blue')
            plt.plot(t_values, failure_risks, label='失败风险', color='orange')
            plt.plot(t_values, total_risks, label='总风险', color='green')
            plt.axvline(mean_time, color='red', linestyle='--', label=f'最优检测时点: {mean_time:.2f}周')
            plt.title(f"BMI分组{cluster} - 风险曲线")
            plt.xlabel("检测孕周 (周)")
            plt.ylabel("风险值")
            plt.legend()
            plt.grid()
            plt.show()
    
    """NIPT最优时点确定器"""

    def __init__(self, lambda_weight=0.8):
        """
        初始化NIPT优化器

        Parameters:
        lambda_weight: float, 权重参数λ，默认0.8
        """
        self.lambda_weight = lambda_weight
        self.data = None
        self.male_data = None
        self.bmi_groups = {}
        self.regression_models = {}
        self.optimal_times = {}

    def load_data(self, data_path):
        """加载数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(data_path, encoding='utf-8')
        print(f"数据加载完成，共{len(self.data)}条记录")

        # 筛选男胎数据（Y染色体浓度不为空）
        self.male_data = self.data[
            (self.data['Y染色体浓度'].notna()) &
            (self.data['检测孕周'] >= 10) &
            (self.data['检测孕周'] <= 25)
        ].copy()
        print(f"有效男胎数据：{len(self.male_data)}条")

    def analyze_bmi_groups(self):
        """分析BMI分组"""
        print("\n=== BMI分组分析 ===")

        # 分析每个BMI聚类
        clusters = sorted(self.male_data['BMI_cluster'].unique())

        for cluster in clusters:
            cluster_data = self.male_data[self.male_data['BMI_cluster'] == cluster]
            bmis = cluster_data['孕妇BMI']

            self.bmi_groups[cluster] = {
                'data': cluster_data,
                'min_bmi': bmis.min(),
                'max_bmi': bmis.max(),
                'mean_bmi': bmis.mean(),
                'count': len(cluster_data)
            }

            print(f"分组{cluster}: BMI[{bmis.min():.1f}, {bmis.max():.1f}], "
                  f"均值={bmis.mean():.1f}, 样本数={len(cluster_data)}")

    def build_regression_models(self):
        """建立Y染色体浓度的非线性混合规划模型"""
        print("\n=== 非线性混合规划模型建立 ===")

        for cluster, group_info in self.bmi_groups.items():
            data = group_info['data']

            if len(data) < 10:
                print(f"分组{cluster}样本数不足，跳过建模")
                continue

            # 准备回归数据
            X = data[['身高', '体重', '年龄', '孕妇BMI']].values
            y = data['Y染色体浓度'].values

            # 定义非线性模型函数
            def nonlinear_model(params, X):
                a, b, c, d, e = params
                height, weight, age, bmi = X.T
                return a + b * height + c * weight + d * np.log(age + 1) + e * np.sqrt(bmi)

            # 定义目标函数（最小化残差平方和）
            def objective(params):
                y_pred = nonlinear_model(params, X)
                residuals = y - y_pred
                return np.sum(residuals**2)

            # 初始参数猜测
            initial_params = [0.1, 0.1, 0.1, 0.1, 0.1]

            # 优化参数
            result = minimize(objective, initial_params, method='L-BFGS-B')

            if result.success:
                optimal_params = result.x
                y_pred = nonlinear_model(optimal_params, X)
                residuals = y - y_pred
                sigma = np.std(residuals, ddof=3)  # 自由度调整

                # 模型评估
                r2 = r2_score(y, y_pred)

                self.regression_models[cluster] = {
                    'params': optimal_params,
                    'sigma': sigma,
                    'r2': r2,
                    'n': len(data)
                }

                print(f"分组{cluster}: 模型参数 = {optimal_params}")
                print(f"         R² = {r2:.4f}, σ = {sigma:.6f}, n = {len(data)}")
            else:
                print(f"分组{cluster}: 优化失败")

    def delay_risk(self, t):
        """延迟风险函数"""
        return max(0, (20**(t-10)-1)/(20**14-1))

    def failure_risk(self, t, bmi, model_params):
        """失败风险函数"""
        # 预测Y染色体浓度
        params = model_params['params']
        predicted_concentration = (
            params[0] +
            params[1] * bmi +
            params[2] * np.log(t + 1) +
            params[3] * np.sqrt(bmi)
        )

        # 计算Z值
        z_score = (0.04 - predicted_concentration) / model_params['sigma']

        # 返回失败概率（正态分布CDF）
        return stats.norm.cdf(z_score)

    def total_risk(self, t, bmi, model_params):
        """总风险函数"""
        delay_r = self.delay_risk(t)
        failure_r = self.failure_risk(t, bmi, model_params)
        return self.lambda_weight * delay_r + (1 - self.lambda_weight) * failure_r

    def find_optimal_timing(self):
        """寻找最优检测时点"""
        print(f"\n=== 最优时点计算 (λ={self.lambda_weight}) ===")

        for cluster, model_params in self.regression_models.items():
            mean_bmi = self.bmi_groups[cluster]['mean_bmi']

            # 优化函数
            def objective(t):
                return self.total_risk(t, mean_bmi, model_params)

            # 在10-25周范围内寻找最优值
            result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
            optimal_time = result.x
            min_risk = result.fun

            # 计算各项风险组件
            delay_r = self.delay_risk(optimal_time)
            failure_r = self.failure_risk(optimal_time, mean_bmi, model_params)

            self.optimal_times[cluster] = {
                'optimal_time': optimal_time,
                'total_risk': min_risk,
                'delay_risk': delay_r,
                'failure_risk': failure_r
            }

            bmi_range = self.bmi_groups[cluster]
            print(f"\nBMI分组{cluster} ({bmi_range['min_bmi']:.1f}-{bmi_range['max_bmi']:.1f}):")
            print(f"  最优检测时点: {optimal_time:.1f}周")
            print(f"  总风险: {min_risk:.4f}")
            print(f"  - 延迟风险: {delay_r:.4f}")
            print(f"  - 失败风险: {failure_r:.4f}")

def main():
    """主函数 - 完整分析流程"""

    # 初始化分析器
    analyzer = NIPTOptimizer(lambda_weight=0.8)

    # 1. 加载数据
    data_path = "处理后的数据_带聚类.csv"  # 请修改为实际路径
    analyzer.load_data(data_path)

    # 2. BMI分组分析
    analyzer.analyze_bmi_groups()

    # 3. 建立非线性混合规划模型
    analyzer.build_regression_models()

    # 4. 寻找最优检测时点
    analyzer.find_optimal_timing()

    # 5. 分析误差对结果的影响
    analyzer.analyze_error_impact(num_simulations=1000)

    print("\n分析完成！")

if __name__ == "__main__":
    main()