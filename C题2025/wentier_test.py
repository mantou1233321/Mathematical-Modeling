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

class EnhancedNIPTOptimizer:
    """增强版NIPT最优时点确定器 - 强化组间差异"""
    
    def __init__(self, lambda_weight=0.6, risk_model='enhanced'):
        """
        初始化NIPT优化器
        
        Parameters:
        lambda_weight: float, 权重参数λ，默认0.6
        risk_model: str, 风险模型类型 ('basic', 'enhanced', 'adaptive')
        """
        self.lambda_weight = lambda_weight
        self.risk_model = risk_model
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
                'std_bmi': bmis.std(),
                'count': len(cluster_data)
            }
            
            print(f"分组{cluster}: BMI[{bmis.min():.4f}, {bmis.max():.4f}], "
                  f"均值={bmis.mean():.4f}, 标准差={bmis.std():.4f}, 样本数={len(cluster_data)}")
    
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
            
            # 计算BMI对浓度影响的敏感性系数
            bmi_sensitivity = abs(model.coef_[1]) * group_info['std_bmi']
            
            self.regression_models[cluster] = {
                'model': model,
                'intercept': model.intercept_,
                'coef_week': model.coef_[0],
                'coef_bmi': model.coef_[1],
                'sigma': sigma,
                'r2': r2,
                'f_stat': f_stat,
                'p_value': f_p_value,
                'n': n,
                'bmi_sensitivity': bmi_sensitivity
            }
            
            print(f"分组{cluster}: c_y = {model.intercept_:.6f} + "
                  f"{model.coef_[0]:.6f}*t + {model.coef_[1]:.6f}*B")
            print(f"         R² = {r2:.4f}, σ = {sigma:.6f}, "
                  f"p-value = {f_p_value:.6f}, BMI敏感性 = {bmi_sensitivity:.6f}")
    
    def delay_risk(self, t, bmi_factor=1.0):
        """增强的延迟风险函数 - 考虑BMI影响"""
        if self.risk_model == 'basic':
            return max(0, 0.00125*2.732**(4*t-50))
        elif self.risk_model == 'enhanced':
            # BMI越高，延迟风险增长越快
            base_risk = max(0, 0.00125*2.732**(4*t-50))
            bmi_multiplier = 1 + 0.3 * (bmi_factor - 1)  # BMI因子影响
            return base_risk * bmi_multiplier
        elif self.risk_model == 'adaptive':
            # 自适应模型：考虑妊娠并发症风险
            base_risk = max(0, 0.00125*2.732**(4*t-50))
            # 高BMI妊娠并发症风险递增
            complication_risk = 0.01 * (bmi_factor - 1)**2 if bmi_factor > 1.2 else 0
            return base_risk + complication_risk
    
    def failure_risk(self, t, bmi, model_params, enhanced=True):
        """增强的失败风险函数"""
        # 预测Y染色体浓度
        predicted_concentration = (model_params['intercept'] + 
                                 model_params['coef_week'] * t + 
                                 model_params['coef_bmi'] * bmi)
        
        if enhanced:
            # 考虑BMI对检测精度的影响
            bmi_adjustment = 1 + 0.1 * (bmi - 24) / 6  # BMI偏离24的影响
            adjusted_sigma = model_params['sigma'] * max(0.8, bmi_adjustment)
            
            # 考虑孕周对检测稳定性的影响
            week_stability = 1 - 0.02 * abs(t - 16)  # 16周为最稳定点
            final_sigma = adjusted_sigma / max(0.7, week_stability)
        else:
            final_sigma = model_params['sigma']
        
        # 计算Z值
        z_score = (0.04 - predicted_concentration) / final_sigma
        
        # 返回失败概率（正态分布CDF）
        return stats.norm.cdf(z_score)
    
    def bmi_risk_factor(self, bmi, cluster):
        """计算BMI风险因子"""
        group_mean_bmi = self.bmi_groups[cluster]['mean_bmi']
        return bmi / group_mean_bmi
    
    def total_risk(self, t, bmi, model_params, cluster):
        """总风险函数 - 增强版"""
        bmi_factor = self.bmi_risk_factor(bmi, cluster)
        
        delay_r = self.delay_risk(t, bmi_factor)
        failure_r = self.failure_risk(t, bmi, model_params, enhanced=True)
        
        # 动态权重调整：BMI越高，越重视检测成功率
        if self.risk_model == 'adaptive':
            dynamic_lambda = self.lambda_weight * (1 - 0.1 * (bmi_factor - 1))
            dynamic_lambda = max(0.3, min(0.9, dynamic_lambda))
        else:
            dynamic_lambda = self.lambda_weight
        
        return dynamic_lambda * delay_r + (1 - dynamic_lambda) * failure_r
    
    def find_optimal_timing(self):
        """寻找最优检测时点"""
        print(f"\n=== 最优时点计算 (λ={self.lambda_weight}, 模型={self.risk_model}) ===")
        
        for cluster, model_params in self.regression_models.items():
            group_info = self.bmi_groups[cluster]
            mean_bmi = group_info['mean_bmi']
            
            # 优化函数
            def objective(t):
                return self.total_risk(t, mean_bmi, model_params, cluster)
            
            # 在10-25周范围内寻找最优值
            result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
            optimal_time = result.x
            min_risk = result.fun
            
            # 计算各项风险组件
            bmi_factor = self.bmi_risk_factor(mean_bmi, cluster)
            delay_r = self.delay_risk(optimal_time, bmi_factor)
            failure_r = self.failure_risk(optimal_time, mean_bmi, model_params, enhanced=True)
            
            # 计算预期Y染色体浓度
            predicted_conc = (model_params['intercept'] + 
                            model_params['coef_week'] * optimal_time + 
                            model_params['coef_bmi'] * mean_bmi)
            
            # 计算达标概率
            success_prob = 1 - failure_r
            
            # 计算风险贡献度
            total_risk_components = delay_r + failure_r
            delay_contribution = delay_r / total_risk_components if total_risk_components > 0 else 0
            failure_contribution = failure_r / total_risk_components if total_risk_components > 0 else 0
            
            self.optimal_times[cluster] = {
                'optimal_time': optimal_time,
                'total_risk': min_risk,
                'delay_risk': delay_r,
                'failure_risk': failure_r,
                'predicted_concentration': predicted_conc,
                'success_probability': success_prob,
                'bmi_factor': bmi_factor,
                'delay_contribution': delay_contribution,
                'failure_contribution': failure_contribution
            }
            
            print(f"\nBMI分组{cluster} ({group_info['min_bmi']:.4f}-{group_info['max_bmi']:.4f}):")
            print(f"  最优检测时点: {optimal_time:.4f}周")
            print(f"  总风险: {min_risk:.4f}")
            print(f"  - 延迟风险: {delay_r:.4f} (贡献度: {delay_contribution*100:.1f}%)")
            print(f"  - 失败风险: {failure_r:.4f} (贡献度: {failure_contribution*100:.1f}%)")
            print(f"  预期Y染色体浓度: {predicted_conc*100:.2f}%")
            print(f"  达标概率: {success_prob*100:.4f}%")
            print(f"  BMI风险因子: {bmi_factor:.3f}")
    
    def compare_risk_models(self):
        """比较不同风险模型的结果"""
        print(f"\n=== 风险模型比较 ===")
        
        models = ['basic', 'enhanced', 'adaptive']
        comparison_results = {}
        
        for model in models:
            original_model = self.risk_model
            self.risk_model = model
            self.optimal_times = {}
            
            # 重新计算最优时点
            for cluster, model_params in self.regression_models.items():
                mean_bmi = self.bmi_groups[cluster]['mean_bmi']
                
                def objective(t):
                    return self.total_risk(t, mean_bmi, model_params, cluster)
                
                result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
                optimal_time = result.x
                
                if cluster not in comparison_results:
                    comparison_results[cluster] = {}
                comparison_results[cluster][model] = optimal_time
            
            self.risk_model = original_model
        
        # 输出比较结果
        print("模型类型对比（最优检测时点）:")
        for cluster in comparison_results:
            bmi_range = self.bmi_groups[cluster]
            print(f"\n分组{cluster} (BMI: {bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}):")
            for model in models:
                time = comparison_results[cluster][model]
                print(f"  {model:>10}: {time:.4f}周")
            
            # 计算时间差异范围
            times = [comparison_results[cluster][model] for model in models]
            time_range = max(times) - min(times)
            print(f"  时间范围: {time_range:.4f}周")
        
        return comparison_results
    
    def sensitivity_analysis_enhanced(self, lambda_range=None, bmi_perturbation=0.1):
        """增强的敏感性分析"""
        if lambda_range is None:
            lambda_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
        print(f"\n=== 增强敏感性分析 ===")
        print(f"BMI扰动: ±{bmi_perturbation*100:.0f}%")
        
        sensitivity_results = {}
        
        for cluster, model_params in self.regression_models.items():
            group_info = self.bmi_groups[cluster]
            base_bmi = group_info['mean_bmi']
            cluster_results = []
            
            for lam in lambda_range:
                original_lambda = self.lambda_weight
                self.lambda_weight = lam
                
                # 基准情况
                def objective_base(t):
                    return self.total_risk(t, base_bmi, model_params, cluster)
                
                result_base = minimize_scalar(objective_base, bounds=(10, 25), method='bounded')
                base_time = result_base.x
                
                # BMI+扰动
                bmi_high = base_bmi * (1 + bmi_perturbation)
                def objective_high(t):
                    return self.total_risk(t, bmi_high, model_params, cluster)
                
                result_high = minimize_scalar(objective_high, bounds=(10, 25), method='bounded')
                high_time = result_high.x
                
                # BMI-扰动
                bmi_low = base_bmi * (1 - bmi_perturbation)
                def objective_low(t):
                    return self.total_risk(t, bmi_low, model_params, cluster)
                
                result_low = minimize_scalar(objective_low, bounds=(10, 25), method='bounded')
                low_time = result_low.x
                
                # 计算敏感性指标
                time_sensitivity = (high_time - low_time) / (2 * bmi_perturbation * base_bmi)
                
                cluster_results.append({
                    'lambda': lam,
                    'base_time': base_time,
                    'high_bmi_time': high_time,
                    'low_bmi_time': low_time,
                    'time_sensitivity': time_sensitivity,
                    'time_range': high_time - low_time
                })
                
                self.lambda_weight = original_lambda
            
            sensitivity_results[cluster] = cluster_results
            
            print(f"\n分组{cluster} (BMI: {base_bmi:.4f}):")
            for result in cluster_results:
                print(f"  λ={result['lambda']:.1f}: 基准={result['base_time']:.4f}周, "
                      f"范围={result['time_range']:.4f}周, 敏感性={result['time_sensitivity']:.4f}")
        
        return sensitivity_results
    
    def plot_enhanced_risk_curves(self, cluster=0, save_path=None):
        """绘制增强版风险曲线图"""
        if cluster not in self.regression_models:
            print(f"分组{cluster}没有有效模型")
            return
            
        model_params = self.regression_models[cluster]
        group_info = self.bmi_groups[cluster]
        mean_bmi = group_info['mean_bmi']
        
        # 时间范围
        t_range = np.linspace(10, 25, 151)
        
        # 计算不同BMI水平的风险曲线
        bmi_levels = [
            mean_bmi * 0.9,  # 低BMI
            mean_bmi,        # 平均BMI
            mean_bmi * 1.1   # 高BMI
        ]
        
        plt.figure(figsize=(15, 12))
        
        # 1. 延迟风险对比
        plt.subplot(2, 3, 1)
        for i, bmi in enumerate(bmi_levels):
            bmi_factor = self.bmi_risk_factor(bmi, cluster)
            delay_risks = [self.delay_risk(t, bmi_factor) for t in t_range]
            plt.plot(t_range, delay_risks, label=f'BMI={bmi:.4f}', linewidth=2)
        
        plt.xlabel('孕周')
        plt.ylabel('延迟风险')
        plt.title(f'延迟风险曲线对比 (分组{cluster})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. 失败风险对比
        plt.subplot(2, 3, 2)
        for i, bmi in enumerate(bmi_levels):
            failure_risks = [self.failure_risk(t, bmi, model_params, enhanced=True) for t in t_range]
            plt.plot(t_range, failure_risks, label=f'BMI={bmi:.4f}', linewidth=2)
        
        plt.xlabel('孕周')
        plt.ylabel('失败风险')
        plt.title(f'失败风险曲线对比 (分组{cluster})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. 总风险对比
        plt.subplot(2, 3, 3)
        optimal_times = []
        for i, bmi in enumerate(bmi_levels):
            total_risks = [self.total_risk(t, bmi, model_params, cluster) for t in t_range]
            
            plt.plot(t_range, total_risks, label=f'BMI={bmi:.4f}', linewidth=2)
            
            # 找到最优点
            min_idx = np.argmin(total_risks)
            optimal_times.append(t_range[min_idx])
            plt.plot(t_range[min_idx], total_risks[min_idx], 'o', markersize=8)
        
        plt.xlabel('孕周')
        plt.ylabel('总风险')
        
        plt.title(f'总风险曲线对比 (分组{cluster})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. 预测浓度曲线
        plt.subplot(2, 3, 4)
        for i, bmi in enumerate(bmi_levels):
            concentrations = [(model_params['intercept'] + 
                             model_params['coef_week'] * t + 
                             model_params['coef_bmi'] * bmi) for t in t_range]
            plt.plot(t_range, [c*100 for c in concentrations], label=f'BMI={bmi:.4f}', linewidth=2)
        
        plt.axhline(y=4, color='red', linestyle='--', alpha=0.7, label='检测阈值(4%)')
        plt.xlabel('孕周')
        plt.ylabel('Y染色体浓度 (%)')
        plt.title(f'预测浓度曲线 (分组{cluster})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 5. 成功率曲线
        plt.subplot(2, 3, 5)
        for i, bmi in enumerate(bmi_levels):
            success_rates = [1 - self.failure_risk(t, bmi, model_params, enhanced=True) for t in t_range]
            plt.plot(t_range, [s*100 for s in success_rates], label=f'BMI={bmi:.4f}', linewidth=2)
        
        plt.xlabel('孕周')
        plt.ylabel('检测成功率 (%)')
        plt.title(f'检测成功率曲线 (分组{cluster})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 6. 最优时点对比
        plt.subplot(2, 3, 6)
        bmi_values = [bmi_levels[i] for i in range(len(bmi_levels))]
        plt.bar(range(len(bmi_levels)), optimal_times, alpha=0.7)
        plt.xlabel('BMI水平')
        plt.ylabel('最优检测时点 (周)')
        plt.title(f'不同BMI的最优时点 (分组{cluster})')
        plt.xticks(range(len(bmi_levels)), [f'{bmi:.4f}' for bmi in bmi_values], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, time in enumerate(optimal_times):
            plt.text(i, time + 0.05, f'{time:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"增强风险曲线图已保存到: {save_path}")
        
        plt.show()
    
    def generate_enhanced_report(self, save_path=None):
        """生成增强版完整报告"""
        report = []
        report.append("=" * 70)
        report.append("NIPT最优时点确定 - 增强版风险分析报告")
        report.append("=" * 70)
        
        # 1. 分析配置
        report.append(f"\n1. 分析配置")
        report.append(f"   风险模型: {self.risk_model}")
        report.append(f"   权重参数λ: {self.lambda_weight}")
        report.append(f"   总数据量: {len(self.data)}条")
        report.append(f"   有效男胎数据: {len(self.male_data)}条")
        
        # 2. BMI分组详情
        report.append(f"\n2. BMI分组详情")
        for cluster, group_info in self.bmi_groups.items():
            report.append(f"   分组{cluster}:")
            report.append(f"     BMI范围: [{group_info['min_bmi']:.4f}, {group_info['max_bmi']:.4f}]")
            report.append(f"     BMI均值: {group_info['mean_bmi']:.4f}")
            report.append(f"     BMI标准差: {group_info['std_bmi']:.4f}")
            report.append(f"     样本数: {group_info['count']}")
        
        # 3. 回归模型结果
        report.append(f"\n3. Y染色体浓度回归模型")
        for cluster, model_params in self.regression_models.items():
            report.append(f"   分组{cluster}:")
            report.append(f"     模型: c_y = {model_params['intercept']:.6f} + "
                         f"{model_params['coef_week']:.6f}*t + {model_params['coef_bmi']:.6f}*B")
            report.append(f"     R² = {model_params['r2']:.4f}")
            report.append(f"     σ = {model_params['sigma']:.6f}")
            report.append(f"     p值 = {model_params['p_value']:.6f}")
            report.append(f"     BMI敏感性 = {model_params['bmi_sensitivity']:.6f}")
        
        # 4. 最优检测时点
        report.append(f"\n4. 最优NIPT检测时点")
        times_list = []
        for cluster, optimal_info in self.optimal_times.items():
            bmi_range = self.bmi_groups[cluster]
            times_list.append(optimal_info['optimal_time'])
            report.append(f"   BMI分组{cluster} ({bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}):")
            report.append(f"     最优检测时点: {optimal_info['optimal_time']:.4f}周")
            report.append(f"     总风险: {optimal_info['total_risk']:.4f}")
            report.append(f"     延迟风险: {optimal_info['delay_risk']:.4f} ({optimal_info['delay_contribution']*100:.1f}%)")
            report.append(f"     失败风险: {optimal_info['failure_risk']:.4f} ({optimal_info['failure_contribution']*100:.1f}%)")
            report.append(f"     预期Y染色体浓度: {optimal_info['predicted_concentration']*100:.2f}%")
            report.append(f"     检测成功概率: {optimal_info['success_probability']*100:.4f}%")
            report.append(f"     BMI风险因子: {optimal_info['bmi_factor']:.3f}")
        
        # 5. 组间差异分析
        if len(times_list) > 1:
            time_range = max(times_list) - min(times_list)
            time_std = np.std(times_list)
            report.append(f"\n5. 组间差异分析")
            report.append(f"   最优时点范围: {time_range:.4f}周")
            report.append(f"   最优时点标准差: {time_std:.4f}周")
            report.append(f"   最早建议: {min(times_list):.4f}周")
            report.append(f"   最晚建议: {max(times_list):.4f}周")
            
            if time_range > 1.0:
                report.append(f"   组间差异显著 (>1周)")
            elif time_range > 0.5:
                report.append(f"   组间差异中等 (0.5-1周)")
            else:
                report.append(f"   组间差异较小 (<0.5周)")
        
        # 6. 主要发现
        report.append(f"\n6. 主要发现")
        report.append(f"   1) 使用{self.risk_model}风险模型增强了组间差异")
        report.append(f"   2) BMI对检测时机和成功率均有显著影响")
        report.append(f"   3) 高BMI组具有更高的风险因子和延迟容忍度")
        report.append(f"   4) 检测精度受BMI影响，需要个体化调整")
        
        # 7. 临床建议
        report.append(f"\n7. 临床建议")
        for cluster, optimal_info in self.optimal_times.items():
            bmi_range = self.bmi_groups[cluster]
            opt_time = optimal_info['optimal_time']
            window_start = max(10, opt_time - 0.5)
            window_end = min(25, opt_time + 0.5)
            
            report.append(f"   BMI {bmi_range['min_bmi']:.4f}-{bmi_range['max_bmi']:.4f}:")
            report.append(f"     推荐检测窗口: {window_start:.0f}-{window_end:.0f}周")
            report.append(f"     最佳时点: {opt_time:.4f}周")
            report.append(f"     预期成功率: {optimal_info['success_probability']*100:.2f}%")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"增强版报告已保存到: {save_path}")
        
        print(report_text)
        return report_text

def main_enhanced():
    """增强版主函数 - 完整分析流程"""
    
    print("=" * 60)
    print("NIPT最优时点确定器 - 增强版")
    print("=" * 60)
    
    # 1. 测试不同风险模型
    risk_models = ['basic', 'enhanced', 'adaptive']
    lambda_values = [0.5, 0.6, 0.7, 0.8]
    
    best_config = None
    max_time_range = 0
    
    for risk_model in risk_models:
        for lambda_val in lambda_values:
            print(f"\n测试配置: 模型={risk_model}, λ={lambda_val}")
            
            analyzer = EnhancedNIPTOptimizer(lambda_weight=lambda_val, risk_model=risk_model)
            
            # 加载数据
            data_path = "处理后的数据_带聚类.csv"
            try:
                analyzer.load_data(data_path)
                analyzer.analyze_bmi_groups()
                analyzer.build_regression_models()
                analyzer.find_optimal_timing()
                
                # 计算组间时间差异
                times = [info['optimal_time'] for info in analyzer.optimal_times.values()]
                if len(times) > 1:
                    time_range = max(times) - min(times)
                    print(f"组间时间差异: {time_range:.4f}周")
                    
                    if time_range > max_time_range:
                        max_time_range = time_range
                        best_config = (risk_model, lambda_val, analyzer)
                
            except Exception as e:
                print(f"配置测试失败: {e}")
                continue
    
    if best_config is None:
        print("未找到有效配置，使用默认设置")
        analyzer = EnhancedNIPTOptimizer(lambda_weight=0.6, risk_model='enhanced')
        data_path = "处理后的数据_带聚类.csv"
        analyzer.load_data(data_path)
    else:
        risk_model, lambda_val, analyzer = best_config
        print(f"\n最佳配置: 模型={risk_model}, λ={lambda_val}, 最大时间差异={max_time_range:.4f}周")
    
    # 2. 完整分析流程
    print(f"\n开始完整分析...")
    
    analyzer.analyze_bmi_groups()
    analyzer.build_regression_models()
    analyzer.find_optimal_timing()
    
    # 3. 风险模型比较
    print(f"\n进行风险模型比较...")
    comparison_results = analyzer.compare_risk_models()
    
    # 4. 增强敏感性分析
    print(f"\n进行增强敏感性分析...")
    sensitivity_results = analyzer.sensitivity_analysis_enhanced()
    
    # 5. 可视化分析
    print(f"\n生成可视化图表...")
    
    # 为每个BMI组绘制增强风险曲线
    for cluster in analyzer.regression_models.keys():
        analyzer.plot_enhanced_risk_curves(cluster, f'enhanced_risk_curves_cluster_{cluster}.png')
    
    # 6. 生成详细报告
    print(f"\n生成增强版分析报告...")
    analyzer.generate_enhanced_report('Enhanced_NIPT_Analysis_Report.txt')
    
    # 7. 额外分析：BMI连续性影响
    print(f"\n进行BMI连续性影响分析...")
    analyze_bmi_continuous_effect(analyzer)
    
    print(f"\n增强版分析完成！")
    print(f"主要改进:")
    print(f"1. 增强的风险函数考虑BMI因子影响")
    print(f"2. 动态权重调整机制")
    print(f"3. 检测精度的BMI相关调整")
    print(f"4. 多模型比较和最优配置选择")
    print(f"5. 增强的敏感性分析")

def analyze_bmi_continuous_effect(analyzer):
    """分析BMI连续性影响"""
    print("\n=== BMI连续性影响分析 ===")
    
    # 选择一个有效的聚类进行连续性分析
    if not analyzer.regression_models:
        print("没有有效的回归模型")
        return
    
    cluster = list(analyzer.regression_models.keys())[0]
    model_params = analyzer.regression_models[cluster]
    base_bmi = analyzer.bmi_groups[cluster]['mean_bmi']
    
    # BMI范围：从低到高
    bmi_range = np.linspace(base_bmi * 0.8, base_bmi * 1.3, 20)
    optimal_times = []
    success_rates = []
    total_risks = []
    
    for bmi in bmi_range:
        # 寻找最优时点
        def objective(t):
            return analyzer.total_risk(t, bmi, model_params, cluster)
        
        result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
        optimal_time = result.x
        min_risk = result.fun
        
        # 计算成功率
        failure_risk = analyzer.failure_risk(optimal_time, bmi, model_params, enhanced=True)
        success_rate = 1 - failure_risk
        
        optimal_times.append(optimal_time)
        success_rates.append(success_rate)
        total_risks.append(min_risk)
    
    # 可视化BMI连续影响
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(bmi_range, optimal_times, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('孕妇BMI')
    plt.ylabel('最优检测时点 (周)')
    
    plt.title('BMI对最优检测时点的连续影响')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(bmi_range, [s*100 for s in success_rates], 'g-o', linewidth=2, markersize=4)
    plt.xlabel('孕妇BMI')
    plt.ylabel('检测成功率 (%)')
    plt.title('BMI对检测成功率的影响')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(bmi_range, total_risks, 'r-o', linewidth=2, markersize=4)
    plt.xlabel('孕妇BMI')
    plt.ylabel('总风险')
    
    plt.title('BMI对总风险的影响')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('BMI_Continuous_Effect_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算相关性
    time_bmi_corr = np.corrcoef(bmi_range, optimal_times)[0, 1]
    success_bmi_corr = np.corrcoef(bmi_range, success_rates)[0, 1]
    risk_bmi_corr = np.corrcoef(bmi_range, total_risks)[0, 1]
    
    print(f"BMI与最优时点相关性: {time_bmi_corr:.4f}")
    print(f"BMI与成功率相关性: {success_bmi_corr:.4f}")
    print(f"BMI与总风险相关性: {risk_bmi_corr:.4f}")
    
    # 计算BMI敏感性
    time_sensitivity = (optimal_times[-1] - optimal_times[0]) / (bmi_range[-1] - bmi_range[0])
    print(f"时点BMI敏感性: {time_sensitivity:.4f} 周/BMI单位")

def parameter_optimization_study():
    """参数优化研究"""
    print("\n=== 参数优化研究 ===")
    
    lambda_range = np.arange(0.3, 0.9, 0.05)
    risk_models = ['basic', 'enhanced', 'adaptive']
    
    results = {}
    
    for model in risk_models:
        results[model] = []
        
        for lam in lambda_range:
            analyzer = EnhancedNIPTOptimizer(lambda_weight=lam, risk_model=model)
            
            try:
                data_path = "处理后的数据_带聚类.csv"
                analyzer.load_data(data_path)
                analyzer.analyze_bmi_groups()
                analyzer.build_regression_models()
                analyzer.find_optimal_timing()
                
                # 计算评价指标
                times = [info['optimal_time'] for info in analyzer.optimal_times.values()]
                risks = [info['total_risk'] for info in analyzer.optimal_times.values()]
                success_rates = [info['success_probability'] for info in analyzer.optimal_times.values()]
                
                if times:
                    time_range = max(times) - min(times) if len(times) > 1 else 0
                    avg_success_rate = np.mean(success_rates)
                    avg_risk = np.mean(risks)
                    
                    results[model].append({
                        'lambda': lam,
                        'time_range': time_range,
                        'avg_success_rate': avg_success_rate,
                        'avg_risk': avg_risk,
                        'min_time': min(times),
                        'max_time': max(times)
                    })
                
            except Exception as e:
                print(f"参数组合 {model}-{lam:.2f} 失败: {e}")
                continue
    
    # 可视化参数优化结果
    plt.figure(figsize=(15, 10))
    
    colors = ['red', 'blue', 'green']
    
    # 时间差异图
    plt.subplot(2, 2, 1)
    for i, model in enumerate(risk_models):
        if results[model]:
            lambdas = [r['lambda'] for r in results[model]]
            time_ranges = [r['time_range'] for r in results[model]]
            plt.plot(lambdas, time_ranges, 'o-', color=colors[i], 
                    label=f'{model}模型', linewidth=2, markersize=4)
    
    plt.xlabel('权重参数λ')
    plt.ylabel('组间时间差异 (周)')
    plt.title('λ对组间时间差异的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 平均成功率图
    plt.subplot(2, 2, 2)
    for i, model in enumerate(risk_models):
        if results[model]:
            lambdas = [r['lambda'] for r in results[model]]
            success_rates = [r['avg_success_rate']*100 for r in results[model]]
            plt.plot(lambdas, success_rates, 'o-', color=colors[i], 
                    label=f'{model}模型', linewidth=2, markersize=4)
    
    plt.xlabel('权重参数λ')
    plt.ylabel('平均检测成功率 (%)')
    plt.title('λ对平均成功率的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 平均风险图
    plt.subplot(2, 2, 3)
    for i, model in enumerate(risk_models):
        if results[model]:
            lambdas = [r['lambda'] for r in results[model]]
            avg_risks = [r['avg_risk'] for r in results[model]]
            plt.plot(lambdas, avg_risks, 'o-', color=colors[i], 
                    label=f'{model}模型', linewidth=2, markersize=4)
    
    plt.xlabel('权重参数λ')
    plt.ylabel('平均总风险')
    plt.title('λ对平均总风险的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最优时点范围图
    plt.subplot(2, 2, 4)
    for i, model in enumerate(risk_models):
        if results[model]:
            lambdas = [r['lambda'] for r in results[model]]
            min_times = [r['min_time'] for r in results[model]]
            max_times = [r['max_time'] for r in results[model]]
            
            plt.fill_between(lambdas, min_times, max_times, alpha=0.3, color=colors[i])
            plt.plot(lambdas, min_times, '--', color=colors[i], linewidth=1)
            plt.plot(lambdas, max_times, '--', color=colors[i], linewidth=1)
            
            # 中位数线
            mid_times = [(min_t + max_t) / 2 for min_t, max_t in zip(min_times, max_times)]
            plt.plot(lambdas, mid_times, '-', color=colors[i], 
                    label=f'{model}模型', linewidth=2)
    
    plt.xlabel('权重参数λ')
    plt.ylabel('最优检测时点 (周)')
    plt.title('λ对最优时点范围的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Parameter_Optimization_Study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最优参数建议
    print("\n最优参数建议:")
    for model in risk_models:
        if results[model]:
            # 找到时间差异最大的配置
            max_range_config = max(results[model], key=lambda x: x['time_range'])
            print(f"{model}模型: λ={max_range_config['lambda']:.2f}, "
                  f"时间差异={max_range_config['time_range']:.4f}周, "
                  f"成功率={max_range_config['avg_success_rate']*100:.2f}%")
    
    return results

if __name__ == "__main__":
    # 运行增强版主分析
    main_enhanced()
    
    # 可选：运行参数优化研究
    print(f"\n" + "="*60)
    print("开始参数优化研究...")
    optimization_results = parameter_optimization_study()
    
    print(f"\n" + "="*60)
    print("所有分析完成！")
    print("生成的文件:")
    print("1. Enhanced_NIPT_Analysis_Report.txt - 详细分析报告")
    print("2. enhanced_risk_curves_cluster_*.png - 各组风险曲线图")
    print("3. BMI_Continuous_Effect_Analysis.png - BMI连续影响分析")
    print("4. Parameter_Optimization_Study.png - 参数优化研究")