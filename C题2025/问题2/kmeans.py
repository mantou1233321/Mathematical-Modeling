import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTOptimizer:
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
            
            print(f"分组{cluster}: BMI[{bmis.min():.4f}, {bmis.max():.4f}], "
                  f"均值={bmis.mean():.4f}, 样本数={len(cluster_data)}")
    
    def build_regression_models(self):
        """建立Y染色体浓度回归模型"""
        print("\n=== 回归模型建立 ===")
        
        for cluster, group_info in self.bmi_groups.items():
            data = group_info['data']
            
            if len(data) < 10:
                print(f"分组{cluster}样本数不足，跳过建模")
                continue
                
            # 准备回归数据
            X = data[['检测孕周', '孕妇BMI']]
            y = data['Y染色体浓度']
            
            # 线性回归
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测和残差分析
            y_pred = model.predict(X)
            residuals = y - y_pred
            sigma = np.std(residuals, ddof=3)  # 自由度调整
            
            # 模型评估
            r2 = r2_score(y, y_pred)
            
            # 显著性检验
            n = len(data)
            p = 3  # 参数个数（截距+孕周+BMI）
            f_stat = r2 * (n - p) / ((1 - r2) * (p - 1))
            f_p_value = 1 - stats.f.cdf(f_stat, p-1, n-p)
            
            self.regression_models[cluster] = {
                'model': model,
                'intercept': model.intercept_,
                'coef_week': model.coef_[0],
                'coef_bmi': model.coef_[1],
                'sigma': sigma,
                'r2': r2,
                'f_stat': f_stat,
                'p_value': f_p_value,
                'n': n
            }
            
            print(f"分组{cluster}: c_y = {model.intercept_:.6f} + "
                  f"{model.coef_[0]:.6f}*t + {model.coef_[1]:.6f}*B")
            print(f"         R² = {r2:.4f}, σ = {sigma:.6f}, "
                  f"p-value = {f_p_value:.6f}, n = {n}")
    
    def delay_risk(self, t):
        """延迟风险函数"""
        return max(0, (20**(t-1)-1)/(20**14-1))
    
    def failure_risk(self, t, bmi, model_params):
        """失败风险函数"""
        # 预测Y染色体浓度
        predicted_concentration = (model_params['intercept'] + 
                                 model_params['coef_week'] * t + 
                                 model_params['coef_bmi'] * bmi)
        
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
            
            # 计算预期Y染色体浓度
            predicted_conc = (model_params['intercept'] + 
                            model_params['coef_week'] * optimal_time + 
                            model_params['coef_bmi'] * mean_bmi)
            
            # 计算达标概率
            success_prob = 1 - failure_r
            
            self.optimal_times[cluster] = {
                'optimal_time': optimal_time,
                'total_risk': min_risk,
                'delay_risk': delay_r,
                'failure_risk': failure_r,
                'predicted_concentration': predicted_conc,
                'success_probability': success_prob
            }
            
            bmi_range = self.bmi_groups[cluster]
            print(f"\nBMI分组{cluster} ({bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}):")
            print(f"  最优检测时点: {optimal_time:.4f}周")
            print(f"  总风险: {min_risk:.4f}")
            print(f"  - 延迟风险: {delay_r:.4f}")
            print(f"  - 失败风险: {failure_r:.4f}")
            print(f"  预期Y染色体浓度: {predicted_conc*100:.2f}%")
            print(f"  达标概率: {success_prob*100:.4f}%")
    
    def sensitivity_analysis(self, lambda_range=None):
        """λ敏感性分析"""
        if lambda_range is None:
            lambda_range = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            
        print(f"\n=== λ敏感性分析 ===")
        
        sensitivity_results = {}
        
        for cluster, model_params in self.regression_models.items():
            mean_bmi = self.bmi_groups[cluster]['mean_bmi']
            cluster_results = []
            
            for lam in lambda_range:
                # 临时改变λ值
                original_lambda = self.lambda_weight
                self.lambda_weight = lam
                
                # 优化
                def objective(t):
                    return self.total_risk(t, mean_bmi, model_params)
                
                result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
                optimal_time = result.x
                min_risk = result.fun
                
                cluster_results.append({
                    'lambda': lam,
                    'optimal_time': optimal_time,
                    'total_risk': min_risk
                })
                
                # 恢复原始λ值
                self.lambda_weight = original_lambda
            
            sensitivity_results[cluster] = cluster_results
            
            print(f"\nBMI分组{cluster}:")
            for result in cluster_results:
                print(f"  λ={result['lambda']:.2f}: {result['optimal_time']:.4f}周, "
                      f"风险={result['total_risk']:.4f}")
        
        return sensitivity_results
    
    def error_impact_analysis(self, error_rates=None):
        """检测误差影响分析"""
        if error_rates is None:
            error_rates = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20%误差
            
        print(f"\n=== 检测误差影响分析 ===")
        
        for cluster, model_params in self.regression_models.items():
            mean_bmi = self.bmi_groups[cluster]['mean_bmi']
            original_optimal = self.optimal_times[cluster]['optimal_time']
            
            print(f"\nBMI分组{cluster}:")
            print(f"  原最优时点: {original_optimal:.4f}周")
            
            for error_rate in error_rates:
                # 创建误差调整模型
                adjusted_params = model_params.copy()
                adjusted_params['sigma'] = model_params['sigma'] * (1 + error_rate)
                
                # 优化
                def objective(t):
                    delay_r = self.delay_risk(t)
                    failure_r = self.failure_risk(t, mean_bmi, adjusted_params)
                    return self.lambda_weight * delay_r + (1 - self.lambda_weight) * failure_r
                
                result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
                adjusted_optimal = result.x
                adjusted_risk = result.fun
                
                time_diff = adjusted_optimal - original_optimal
                risk_diff = adjusted_risk - self.optimal_times[cluster]['total_risk']
                
                print(f"  误差+{error_rate*100:.0f}%: {adjusted_optimal:.4f}周, "
                      f"时点变化{time_diff:+.4f}周, 风险变化{risk_diff:+.4f}")
    
    def plot_risk_curves(self, cluster=0, save_path=None):
        """绘制风险曲线图"""
        if cluster not in self.regression_models:
            print(f"分组{cluster}没有有效模型")
            return
            
        model_params = self.regression_models[cluster]
        mean_bmi = self.bmi_groups[cluster]['mean_bmi']
        
        # 时间范围
        t_range = np.linspace(10, 25, 151)
        
        # 计算各种风险
        delay_risks = [self.delay_risk(t) for t in t_range]
        failure_risks = [self.failure_risk(t, mean_bmi, model_params) for t in t_range]
        total_risks = [self.total_risk(t, mean_bmi, model_params) for t in t_range]
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t_range, delay_risks, 'r-', label='延迟风险', linewidth=2)
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.ylim(-0.02,0.1)
        plt.title(f'延迟风险曲线 (BMI分组{cluster})')
        plt.grid(True, alpha=0.3)
        
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(t_range, failure_risks, 'b-', label='失败风险', linewidth=2)
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        
        plt.title(f'失败风险曲线 (BMI分组{cluster})')
        plt.grid(True, alpha=0.3)
        
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(t_range, total_risks, 'g-', label='总风险', linewidth=2)
        optimal_time = self.optimal_times[cluster]['optimal_time']
        optimal_risk = self.optimal_times[cluster]['total_risk']
        plt.plot(optimal_time, optimal_risk, 'ro', markersize=8, label=f'最优点({optimal_time:.4f}周)')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        
        plt.title(f'总风险曲线 (BMI分组{cluster}, λ={self.lambda_weight})')
        plt.grid(True, alpha=0.3)
        
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(t_range, delay_risks, 'r-', label='延迟风险', alpha=0.7)
        plt.plot(t_range, failure_risks, 'b-', label='失败风险', alpha=0.7)
        plt.plot(t_range, total_risks, 'g-', label='总风险', linewidth=2)
        plt.plot(optimal_time, optimal_risk, 'ro', markersize=8, label=f'最优点')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.title(f'风险曲线对比 (BMI分组{cluster})')
        plt.grid(True, alpha=0.3)
        
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"风险曲线图已保存到: {save_path}")
        
        plt.show()
    
    def plot_lambda_sensitivity(self, save_path=None):
        """绘制λ敏感性分析图"""
        lambda_range = np.arange(0.3, 1.0, 0.05)
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (cluster, model_params) in enumerate(self.regression_models.items()):
            mean_bmi = self.bmi_groups[cluster]['mean_bmi']
            optimal_times = []
            
            for lam in lambda_range:
                original_lambda = self.lambda_weight
                self.lambda_weight = lam
                
                def objective(t):
                    return self.total_risk(t, mean_bmi, model_params)
                
                result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
                optimal_times.append(result.x)
                
                self.lambda_weight = original_lambda
            
            bmi_range = self.bmi_groups[cluster]
            plt.plot(lambda_range, optimal_times, 'o-', color=colors[i], 
                    label=f'分组{cluster} (BMI:{bmi_range["min_bmi"]:.4f}-{bmi_range["max_bmi"]:.4f})',
                    linewidth=2, markersize=4)
        
        plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.7, label='推荐值λ=0.8')
        plt.xlabel('权重参数λ')
        plt.ylabel('最优检测时点(周)')
        plt.title('λ值对最优检测时点的影响')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"λ敏感性分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """生成完整报告"""
        report = []
        report.append("=" * 60)
        report.append("NIPT最优时点确定 - 问题2完整解答报告")
        report.append("=" * 60)
        
        # 1. 数据概况
        report.append(f"\n1. 数据概况")
        report.append(f"   总数据量: {len(self.data)}条")
        report.append(f"   有效男胎数据: {len(self.male_data)}条")
        report.append(f"   权重参数λ: {self.lambda_weight}")
        
        # 2. BMI分组结果
        report.append(f"\n2. BMI分组结果")
        for cluster, group_info in self.bmi_groups.items():
            report.append(f"   分组{cluster}: BMI[{group_info['min_bmi']:.4f}, {group_info['max_bmi']:.4f}], "
                         f"均值={group_info['mean_bmi']:.4f}, 样本数={group_info['count']}")
        
        # 3. 回归模型
        report.append(f"\n3. Y染色体浓度回归模型")
        for cluster, model_params in self.regression_models.items():
            report.append(f"   分组{cluster}: c_y = {model_params['intercept']:.6f} + "
                         f"{model_params['coef_week']:.6f}*t + {model_params['coef_bmi']:.6f}*B")
            report.append(f"           R² = {model_params['r2']:.4f}, σ = {model_params['sigma']:.6f}, "
                         f"p = {model_params['p_value']:.6f}")
        
        # 4. 最优时点
        report.append(f"\n4. 最优NIPT检测时点")
        for cluster, optimal_info in self.optimal_times.items():
            bmi_range = self.bmi_groups[cluster]
            report.append(f"   BMI分组{cluster} ({bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}):")
            report.append(f"     最优检测时点: {optimal_info['optimal_time']:.4f}周")
            report.append(f"     总风险: {optimal_info['total_risk']:.4f}")
            report.append(f"     延迟风险: {optimal_info['delay_risk']:.4f}")
            report.append(f"     失败风险: {optimal_info['failure_risk']:.4f}")
            report.append(f"     预期Y染色体浓度: {optimal_info['predicted_concentration']*100:.2f}%")
            report.append(f"     达标概率: {optimal_info['success_probability']*100:.4f}%")
        
        # 5. 主要结论
        report.append(f"\n5. 主要结论")
        report.append(f"   1) BMI越高，最优检测时点越晚")
        report.append(f"   2) 高BMI组具有更高的检测成功率")
        report.append(f"   3) 延迟风险和失败风险需要平衡考虑")
        report.append(f"   4) 推荐λ=0.8作为标准权重参数")
        
        # 6. 实践建议
        report.append(f"\n6. 实践建议")
        for cluster, optimal_info in self.optimal_times.items():
            bmi_range = self.bmi_groups[cluster]
            time_range = f"{optimal_info['optimal_time']-0.5:.0f}-{optimal_info['optimal_time']+0.5:.0f}"
            report.append(f"   BMI {bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}: "
                         f"建议{time_range}周检测")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        
        print(report_text)
        return report_text

def main():
    """主函数 - 完整分析流程"""
    
    # 初始化分析器
    analyzer = NIPTOptimizer(lambda_weight=0.8)
    
    # 1. 加载数据
    data_path = "处理后的数据_带聚类.csv"  # 请修改为实际路径
    analyzer.load_data(data_path)
    
    # 2. BMI分组分析
    analyzer.analyze_bmi_groups()
    
    # 3. 建立回归模型
    analyzer.build_regression_models()
    
    # 4. 寻找最优时点
    analyzer.find_optimal_timing()
    
    # 5. 敏感性分析
    analyzer.sensitivity_analysis()
    
    # 6. 误差影响分析
    analyzer.error_impact_analysis()
    
    # 7. 可视化分析
    print("\n正在生成可视化图表...")
    
    # 为每个BMI组绘制风险曲线
    for cluster in analyzer.regression_models.keys():
        analyzer.plot_risk_curves(cluster, f'risk_curves_cluster_{cluster}.png')
    
    # λ敏感性分析图
    analyzer.plot_lambda_sensitivity('lambda_sensitivity.png')
    
    # 8. 生成完整报告
    analyzer.generate_report('NIPT_optimization_report.txt')
    
    print("\n分析完成！所有结果已保存。")

# 额外功能函数
def lambda_optimization_study(data_path):
    """λ值优化研究"""
    print("λ值优化研究")
    print("=" * 40)
    
    lambda_candidates = [0.2,0.3,0.4,0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results_summary = []
    
    for lam in lambda_candidates:
        analyzer = NIPTOptimizer(lambda_weight=lam)
        analyzer.load_data(data_path)
        analyzer.analyze_bmi_groups()
        analyzer.build_regression_models()
        analyzer.find_optimal_timing()
        
        # 收集结果
        avg_time = np.mean([info['optimal_time'] for info in analyzer.optimal_times.values()])
        avg_risk = np.mean([info['total_risk'] for info in analyzer.optimal_times.values()])
        avg_success = np.mean([info['success_probability'] for info in analyzer.optimal_times.values()])
        
        results_summary.append({
            'lambda': lam,
            'avg_optimal_time': avg_time,
            'avg_total_risk': avg_risk,
            'avg_success_rate': avg_success
        })
        
        print(f"λ={lam}: 平均最优时点={avg_time:.5f}周, 平均成功率={avg_success*100:.5f}%")
    
    return results_summary

def cost_benefit_analysis():
    """成本效益分析"""
    print("\n成本效益分析")
    print("=" * 40)
    
    # 成本参数
    delay_cost = 300000  # 延迟发现成本(元)
    fail_cost = 3000     # 检测失败成本(元)
    
    # 计算最优λ
    optimal_lambda = delay_cost / (delay_cost + fail_cost)
    
    print(f"延迟发现成本: {delay_cost:,}元")
    print(f"检测失败成本: {fail_cost:,}元")
    print(f"成本比例: {delay_cost/fail_cost:.0f}:1")
    print(f"理论最优λ值: {optimal_lambda:.4f}")
    print(f"实践推荐λ值: 0.80 (考虑操作性调整)")

if __name__ == "__main__":
    # 运行主分析
    main()
    
