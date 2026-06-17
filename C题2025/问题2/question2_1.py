import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTOptimalTiming:
    """NIPT最优时点计算类"""
    
    def __init__(self, lambda_weight=0.8):
        """
        初始化
        
        Parameters:
        lambda_weight: float, 权重参数λ，默认0.8
        """
        self.lambda_weight = lambda_weight
        
        # BMI分组信息
        self.bmi_groups = {
            0: {'range': [30.6, 33.0], 'mean_bmi': 31.7, 'name': '中等偏高BMI'},
            1: {'range': [33.0, 36.1], 'mean_bmi': 34.2, 'name': '高BMI'},
            2: {'range': [27.0, 30.6], 'mean_bmi': 29.5, 'name': '正常偏高BMI'},
            3: {'range': [36.1, 43.9], 'mean_bmi': 37.9, 'name': '极高BMI'}
        }
        
        # 各分组回归模型参数
        self.regression_models = {
            0: {'beta0': 0.024308, 'beta1': 0.000830, 'beta2': 0.001265, 'sigma': 0.030038, 'r2': 0.127},
            1: {'beta0': 0.113729, 'beta1': 0.001092, 'beta2': -0.001598, 'sigma': 0.031746, 'r2': 0.089},
            2: {'beta0': 0.032771, 'beta1': 0.001579, 'beta2': 0.000873, 'sigma': 0.035863, 'r2': 0.156},
            3: {'beta0': -0.026214, 'beta1': 0.002588, 'beta2': 0.000994, 'sigma': 0.027109, 'r2': 0.201}
        }
    
    def delay_risk(self, t):
        """
        延迟发现风险函数
        R_late(t) = max{0, (t-12)/15}
        """
        return max(0, (t - 12) / 15)
    
    def predict_y_concentration(self, t, bmi, group):
        """
        预测Y染色体浓度
        c_y = β₀ + β₁·t + β₂·B
        """
        model = self.regression_models[group]
        return model['beta0'] + model['beta1'] * t + model['beta2'] * bmi
    
    def failure_risk(self, t, bmi, group):
        """
        检测失败风险函数
        R_fail(t,B) = P(c_y < 0.04) = Φ((0.04 - f(t,B))/σ)
        """
        model = self.regression_models[group]
        predicted_conc = self.predict_y_concentration(t, bmi, group)
        z_score = (0.04 - predicted_conc) / model['sigma']
        return stats.norm.cdf(z_score)
    
    def total_risk(self, t, bmi, group):
        """
        总风险函数
        R_total(t,B) = λ·R_late(t) + (1-λ)·R_fail(t,B)
        """
        delay_r = self.delay_risk(t)
        failure_r = self.failure_risk(t, bmi, group)
        return self.lambda_weight * delay_r + (1 - self.lambda_weight) * failure_r
    
    def find_optimal_timing(self, group):
        """
        寻找最优检测时点
        """
        mean_bmi = self.bmi_groups[group]['mean_bmi']
        
        # 定义目标函数
        def objective(t):
            return self.total_risk(t, mean_bmi, group)
        
        # 在10-25周范围内优化
        result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
        
        optimal_time = result.x
        min_risk = result.fun
        
        # 计算各项风险组件
        delay_r = self.delay_risk(optimal_time)
        failure_r = self.failure_risk(optimal_time, mean_bmi, group)
        
        # 计算预期Y染色体浓度
        predicted_conc = self.predict_y_concentration(optimal_time, mean_bmi, group)
        
        # 计算达标概率
        success_prob = 1 - failure_r
        
        return {
            'group': group,
            'bmi_range': f"[{self.bmi_groups[group]['range'][0]}-{self.bmi_groups[group]['range'][1]}]",
            'optimal_time': round(optimal_time, 1),
            'total_risk': round(min_risk, 4),
            'delay_risk': round(delay_r, 4),
            'failure_risk': round(failure_r, 4),
            'predicted_concentration': round(predicted_conc * 100, 2),
            'success_probability': round(success_prob * 100, 1)
        }
    
    def calculate_all_optimal_timings(self):
        """
        计算所有BMI组的最优时点
        """
        results = []
        
        print("各BMI分组的最优检测时点及风险分析:")
        print("=" * 80)
        
        for group in range(4):
            result = self.find_optimal_timing(group)
            results.append(result)
            
            print(f"\nBMI分组{group} ({result['bmi_range']}):")
            print(f"  最优检测时点: {result['optimal_time']}周")
            print(f"  总风险: {result['total_risk']}")
            print(f"  - 延迟风险: {result['delay_risk']}")
            print(f"  - 失败风险: {result['failure_risk']}")
            print(f"  预期Y染色体浓度: {result['predicted_concentration']}%")
            print(f"  达标概率: {result['success_probability']}%")
        
        return results
    
    def create_results_table(self, results):
        """
        创建结果表格
        """
        df = pd.DataFrame(results)
        df.columns = ['BMI分组', 'BMI范围', '最优时点(周)', '总风险', '延迟风险', 
                     '失败风险', '预期Y浓度(%)', '达标概率(%)']
        
        print("\n" + "=" * 80)
        print("最优检测时点汇总表:")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    
    def plot_risk_curves(self, group, save_path=None):
        """
        绘制指定BMI组的风险曲线
        """
        mean_bmi = self.bmi_groups[group]['mean_bmi']
        t_range = np.linspace(10, 25, 151)
        
        # 计算各种风险
        delay_risks = [self.delay_risk(t) for t in t_range]
        failure_risks = [self.failure_risk(t, mean_bmi, group) for t in t_range]
        total_risks = [self.total_risk(t, mean_bmi, group) for t in t_range]
        
        # 找到最优点
        optimal_result = self.find_optimal_timing(group)
        optimal_t = optimal_result['optimal_time']
        optimal_risk = optimal_result['total_risk']
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t_range, delay_risks, 'r-', linewidth=2, label='延迟风险')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.title(f'延迟风险曲线 (BMI分组{group})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(t_range, failure_risks, 'b-', linewidth=2, label='失败风险')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.title(f'失败风险曲线 (BMI分组{group})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(t_range, total_risks, 'g-', linewidth=2, label='总风险')
        plt.plot(optimal_t, optimal_risk, 'ro', markersize=8, label=f'最优点({optimal_t}周)')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.title(f'总风险曲线 (BMI分组{group}, λ={self.lambda_weight})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(t_range, delay_risks, 'r-', alpha=0.7, label='延迟风险')
        plt.plot(t_range, failure_risks, 'b-', alpha=0.7, label='失败风险')
        plt.plot(t_range, total_risks, 'g-', linewidth=2, label='总风险')
        plt.plot(optimal_t, optimal_risk, 'ro', markersize=8, label='最优点')
        plt.xlabel('孕周')
        plt.ylabel('风险值')
        plt.title(f'风险曲线对比 (BMI分组{group})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"风险曲线图已保存到: {save_path}")
        
        plt.show()
    
    def sensitivity_analysis(self, lambda_range=None):
        """
        λ敏感性分析
        """
        if lambda_range is None:
            lambda_range = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        print("\n" + "=" * 80)
        print("λ敏感性分析:")
        print("=" * 80)
        
        sensitivity_results = {}
        
        for group in range(4):
            mean_bmi = self.bmi_groups[group]['mean_bmi']
            group_results = []
            
            print(f"\nBMI分组{group}:")
            
            for lam in lambda_range:
                # 临时改变λ值
                original_lambda = self.lambda_weight
                self.lambda_weight = lam
                
                # 计算最优时点
                result = self.find_optimal_timing(group)
                optimal_time = result['optimal_time']
                total_risk = result['total_risk']
                
                group_results.append({
                    'lambda': lam,
                    'optimal_time': optimal_time,
                    'total_risk': total_risk
                })
                
                print(f"  λ={lam}: {optimal_time}周, 风险={total_risk}")
                
                # 恢复原始λ值
                self.lambda_weight = original_lambda
            
            sensitivity_results[group] = group_results
        
        return sensitivity_results
    
    def error_impact_analysis(self, error_rates=None):
        """
        检测误差影响分析
        """
        if error_rates is None:
            error_rates = [0.05, 0.1, 0.15, 0.2]
        
        print("\n" + "=" * 80)
        print("检测误差影响分析:")
        print("=" * 80)
        
        error_results = []
        
        for group in range(4):
            # 获取原始最优时点
            original_result = self.find_optimal_timing(group)
            original_optimal = original_result['optimal_time']
            original_risk = original_result['total_risk']
            
            print(f"\nBMI分组{group}:")
            print(f"  原最优时点: {original_optimal}周")
            
            group_error_results = {'group': group, 'original_time': original_optimal}
            
            for error_rate in error_rates:
                # 创建误差调整模型
                original_sigma = self.regression_models[group]['sigma']
                self.regression_models[group]['sigma'] = original_sigma * (1 + error_rate)
                
                # 计算调整后的最优时点
                adjusted_result = self.find_optimal_timing(group)
                adjusted_optimal = adjusted_result['optimal_time']
                adjusted_risk = adjusted_result['total_risk']
                
                time_diff = adjusted_optimal - original_optimal
                risk_diff = adjusted_risk - original_risk
                
                print(f"  误差+{error_rate*100:.0f}%: {adjusted_optimal}周, "
                      f"时点变化{time_diff:+.1f}周, 风险变化{risk_diff:+.4f}")
                
                group_error_results[f'error_{int(error_rate*100)}'] = {
                    'adjusted_time': adjusted_optimal,
                    'time_diff': time_diff,
                    'risk_diff': risk_diff
                }
                
                # 恢复原始sigma
                self.regression_models[group]['sigma'] = original_sigma
            
            error_results.append(group_error_results)
        
        return error_results

def main():
    """
    主函数 - 运行完整分析
    """
    print("NIPT最优检测时点分析")
    print("=" * 50)
    
    # 创建分析器
    analyzer = NIPTOptimalTiming(lambda_weight=0.8)
    
    # 1. 计算各BMI组最优时点
    results = analyzer.calculate_all_optimal_timings()
    
    # 2. 创建结果表格
    results_df = analyzer.create_results_table(results)
    
    # 3. 绘制风险曲线（可选择特定分组）
    print("\n正在绘制BMI分组0的风险曲线...")
    analyzer.plot_risk_curves(group=0, save_path='risk_curves_group0.png')
    
    # 4. λ敏感性分析
    sensitivity_results = analyzer.sensitivity_analysis()
    
    # 5. 检测误差影响分析
    error_results = analyzer.error_impact_analysis()
    
    print("\n分析完成！")
    
    return results_df, sensitivity_results, error_results

# 如果直接运行此脚本
if __name__ == "__main__":
    # 运行主分析
    results_table, sensitivity_data, error_data = main()
    
    # 可以进一步处理结果
    print("\n最终结果表格:")
    print(results_table)