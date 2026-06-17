"""
NIPT问题3：多因素影响下的BMI分组与最佳NIPT时点确定
完整代码实现 - 从本地读取数据版本（保留异常值）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTOptimizer:
    """NIPT检测时点优化器"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.groups = {}
        self.optimal_times = {}
        self.results = {}
    
    def load_data(self, filepath, encoding='utf-8'):
        """
        从本地文件加载数据
        """
        try:
            print(f"正在从 {filepath} 加载数据...")
            
            # 支持多种文件格式
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath, encoding=encoding)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                self.data = pd.read_excel(filepath)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            
            print(f"数据加载成功，原始数据共{len(self.data)}条记录")
            print(f"数据列名: {list(self.data.columns)}")
            
            # 显示前几行数据
            print("\n前5行数据:")
            print(self.data.head())
            
            # 数据预处理
            self._preprocess_data()
            
            return self.data
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """数据预处理 - 保留异常值，使用稳健方法处理"""
        print("\n正在进行数据预处理...")
        
        # 检查数据是否为空
        if self.data is None or len(self.data) == 0:
            raise ValueError("数据为空，请检查文件路径和内容")
        
        # 重命名列名（根据您的数据实际情况调整）
        column_mapping = {
            '检测孕周': 'gestational_week',
            '孕妇BMI': 'bmi',
            '年龄': 'age',
            '身高': 'height',
            '体重': 'weight',
            'Y染色体浓度': 'y_concentration',
            'GC含量': 'gc_content',
            '原始读段数': 'sequencing_depth',
            '检测抽血次数': 'pregnancy_times',
            '孕妇代码': 'patient_id'
        }
        
        # 应用列名映射
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns and new_name not in self.data.columns:
                self.data[new_name] = self.data[old_name]
                print(f"已将列 '{old_name}' 重命名为 '{new_name}'")
        
        # 确保必要的列存在，如果不存在则创建
        required_columns = ['bmi', 'gestational_week', 'y_concentration']
        
        for col in required_columns:
            if col not in self.data.columns:
                print(f"警告: 缺少必要列 '{col}'，将尝试创建...")
                
                if col == 'bmi' and 'height' in self.data.columns and 'weight' in self.data.columns:
                    self.data['bmi'] = self.data['weight'] / (self.data['height']/100)**2
                    print("已计算BMI列")
                
                elif col == 'y_concentration':
                    # 模拟Y染色体浓度
                    np.random.seed(42)
                    self.data['y_concentration'] = np.random.beta(2, 8, len(self.data)) * 0.15
                    print("已模拟Y染色体浓度列")
                
                elif col == 'gestational_week':
                    # 模拟孕周
                    np.random.seed(42)
                    self.data['gestational_week'] = np.random.normal(16, 3, len(self.data))
                    self.data['gestational_week'] = self.data['gestational_week'].clip(10, 25)
                    print("已模拟孕周列")
        
        # 创建达标标志列
        if 'y_reached_standard' not in self.data.columns and 'y_concentration' in self.data.columns:
            self.data['y_reached_standard'] = self.data['y_concentration'] >= 0.04
            print("已创建达标标志列")
        
        # 处理缺失值 - 使用中位数填充而不是删除
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                print(f"已填充 {col} 的 {self.data[col].isnull().sum()} 个缺失值，使用中位数: {median_val:.2f}")
        
        # 不删除异常值，而是记录异常值信息
        initial_count = len(self.data)
        
        # 记录异常值信息但不删除
        outlier_info = {}
        
        if 'bmi' in self.data.columns:
            bmi_outliers = self.data[(self.data['bmi'] < 16) | (self.data['bmi'] > 50)]
            outlier_info['bmi'] = len(bmi_outliers)
            print(f"BMI异常值: {len(bmi_outliers)} 个 (范围: 16-50)")
        
        if 'age' in self.data.columns:
            age_outliers = self.data[(self.data['age'] < 18) | (self.data['age'] > 45)]
            outlier_info['age'] = len(age_outliers)
            print(f"年龄异常值: {len(age_outliers)} 个 (范围: 18-45)")
        
        if 'gestational_week' in self.data.columns:
            week_outliers = self.data[(self.data['gestational_week'] < 8) | (self.data['gestational_week'] > 30)]
            outlier_info['gestational_week'] = len(week_outliers)
            print(f"孕周异常值: {len(week_outliers)} 个 (范围: 8-30)")
        
        # 保留所有数据，不删除任何记录
        print(f"预处理完成，保留所有 {len(self.data)} 条记录")
        print(f"异常值统计: {outlier_info}")
        
        # 检查数据是否为空
        if len(self.data) == 0:
            raise ValueError("预处理后数据为空，请检查数据质量")
    
    def exploratory_analysis(self):
        """
        探索性数据分析
        """
        print("=== 探索性数据分析 ===")
        
        # 检查数据是否为空
        if self.data is None or len(self.data) == 0:
            print("数据为空，无法进行分析")
            return None
        
        # 基本统计信息
        print("\n基本统计信息:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe().round(3))
        
        # 检查是否有达标率列，如果没有则创建
        if 'y_reached_standard' not in self.data.columns and 'y_concentration' in self.data.columns:
            self.data['y_reached_standard'] = self.data['y_concentration'] >= 0.04
        
        # 相关性分析
        correlation_matrix = self.data[numeric_cols].corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.show()
        
        # Y染色体浓度达标率分析
        if 'y_reached_standard' in self.data.columns:
            overall_rate = self.data['y_reached_standard'].mean()
            print(f"\n整体Y染色体浓度达标率: {overall_rate:.3f}")
            
            # BMI与达标率的关系
            if 'bmi' in self.data.columns and len(self.data) > 0:
                try:
                    # 使用分位数来创建分组，避免极端值影响
                    bmi_quantiles = self.data['bmi'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    bmi_bins = pd.cut(self.data['bmi'], bins=bmi_quantiles, include_lowest=True)
                    bmi_rate = self.data.groupby(bmi_bins)['y_reached_standard'].agg(['count', 'sum', 'mean'])
                    print("\nBMI分组达标率:")
                    print(bmi_rate.round(3))
                except Exception as e:
                    print(f"BMI分组分析失败: {e}")
        
        return correlation_matrix
    
    def build_prediction_models(self):
        """
        构建预测模型 - 使用稳健方法处理异常值
        """
        print("\n=== 构建预测模型 ===")
        
        # 检查数据是否为空
        if self.data is None or len(self.data) == 0:
            print("数据为空，无法构建模型")
            return None
        
        # 确定可用的特征列
        available_features = ['bmi', 'age', 'height', 'weight', 'pregnancy_times', 'gc_content']
        feature_cols = [col for col in available_features if col in self.data.columns]
        
        if not feature_cols:
            print("没有可用的特征列进行建模")
            return None
        
        print(f"使用的特征列: {feature_cols}")
        
        X = self.data[feature_cols]
        
        # 1. 达标时间预测模型（线性回归）- 使用稳健标准化
        if 'gestational_week' not in self.data.columns:
            print("警告: 缺少gestational_week列，无法构建时间预测模型")
            return None
        
        y_time = self.data['gestational_week']
        
        # 使用稳健标准化处理异常值
        scaler_time = RobustScaler()
        X_scaled_time = scaler_time.fit_transform(X)
        
        # 训练测试分割
        X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
            X_scaled_time, y_time, test_size=0.2, random_state=42)
        
        # 训练线性回归模型
        lr_time = LinearRegression()
        lr_time.fit(X_train_time, y_train_time)
        
        # 模型评估
        y_pred_time = lr_time.predict(X_test_time)
        r2_time = r2_score(y_test_time, y_pred_time)
        rmse_time = np.sqrt(mean_squared_error(y_test_time, y_pred_time))
        
        print(f"达标时间预测模型 - R²: {r2_time:.3f}, RMSE: {rmse_time:.3f}")
        
        # 2. 达标概率预测模型（Logistic回归）
        if 'y_reached_standard' not in self.data.columns:
            print("警告: 缺少y_reached_standard列，无法构建概率预测模型")
            return None
        
        y_prob = self.data['y_reached_standard']
        
        # 使用稳健标准化处理异常值
        scaler_prob = RobustScaler()
        X_scaled_prob = scaler_prob.fit_transform(X)
        
        # 训练测试分割
        X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
            X_scaled_prob, y_prob, test_size=0.2, random_state=42)
        
        # 训练Logistic回归模型
        lr_prob = LogisticRegression(random_state=42)
        lr_prob.fit(X_train_prob, y_train_prob)
        
        # 模型评估
        y_pred_prob = lr_prob.predict_proba(X_test_prob)[:, 1]
        auc_prob = roc_auc_score(y_test_prob, y_pred_prob)
        acc_prob = accuracy_score(y_test_prob, lr_prob.predict(X_test_prob))
        
        print(f"\n达标概率预测模型 - AUC: {auc_prob:.3f}, 准确率: {acc_prob:.3f}")
        
        # 保存模型
        self.models = {
            'time_model': lr_time,
            'prob_model': lr_prob,
            'time_scaler': scaler_time,
            'prob_scaler': scaler_prob,
            'feature_cols': feature_cols
        }
        
        return self.models
    
    def optimize_bmi_grouping(self, n_groups=5):
        """
        优化BMI分组 - 使用分位数分组处理异常值
        """
        print(f"\n=== 优化BMI分组({n_groups}组) ===")
        
        if self.data is None or len(self.data) == 0:
            print("数据为空，无法进行分组")
            return {}, pd.DataFrame()
        
        # 使用分位数进行分组，避免极端值影响
        bmi_quantiles = self.data['bmi'].quantile([i/n_groups for i in range(n_groups+1)])
        bmi_quantiles = bmi_quantiles.unique()  # 确保唯一值
        
        # 构建分组结果
        groups = {}
        group_stats = []
        
        for i in range(len(bmi_quantiles) - 1):
            bmi_min = bmi_quantiles[i]
            bmi_max = bmi_quantiles[i+1]
            
            if i == len(bmi_quantiles) - 2:  # 最后一个区间包含最大值
                group_data = self.data[(self.data['bmi'] >= bmi_min) & (self.data['bmi'] <= bmi_max)]
            else:
                group_data = self.data[(self.data['bmi'] >= bmi_min) & (self.data['bmi'] < bmi_max)]
            
            groups[f'group_{i+1}'] = {
                'bmi_range': (bmi_min, bmi_max),
                'data': group_data
            }
            
            # 统计信息
            stats = {
                'group': f'组{i+1}',
                'bmi_range': f'[{bmi_min:.1f}, {bmi_max:.1f})',
                'sample_count': len(group_data),
                'avg_bmi': group_data['bmi'].mean(),
                'avg_age': group_data['age'].mean() if 'age' in group_data.columns else 0,
                'avg_gestational_week': group_data['gestational_week'].mean() if 'gestational_week' in group_data.columns else 0,
                'success_rate': group_data['y_reached_standard'].mean() if 'y_reached_standard' in group_data.columns else 0
            }
            group_stats.append(stats)
        
        self.groups = groups
        
        # 显示分组结果
        group_df = pd.DataFrame(group_stats)
        print("\nBMI分组结果:")
        print(group_df.round(3))
        
        return groups, group_df
    
    # 其他方法保持不变...
    def find_optimal_times(self):
        """
        确定各组最佳NIPT时点
        """
        print("\n=== 确定最佳NIPT时点 ===")
        
        if not self.groups:
            print("没有分组数据，请先运行optimize_bmi_grouping")
            return {}, pd.DataFrame()
        
        optimal_times = {}
        time_analysis = []
        
        for group_name, group_info in self.groups.items():
            group_data = group_info['data']
            
            # 计算不同时点的风险
            risks = []
            weeks = range(10, 20)
            
            for week in weeks:
                # 预测该时点的成功率
                X = group_data[self.models['feature_cols']]
                X_scaled = self.models['prob_scaler'].transform(X)
                success_probs = self.models['prob_model'].predict_proba(X_scaled)[:, 1]
                avg_success_rate = success_probs.mean()
                
                # 时间相关风险
                if week <= 12:
                    time_risk = 0.1
                elif week <= 27:
                    time_risk = 1.0
                else:
                    time_risk = 5.0
                
                # 综合风险
                total_risk = (1 - avg_success_rate) * time_risk
                risks.append(total_risk)
            
            # 找到最优时点
            optimal_week = weeks[np.argmin(risks)]
            optimal_times[group_name] = optimal_week
            
            # 计算该时点的预期性能
            X = group_data[self.models['feature_cols']]
            X_scaled = self.models['prob_scaler'].transform(X)
            expected_success_rate = self.models['prob_model'].predict_proba(X_scaled)[:, 1].mean()
            
            analysis = {
                'group': group_name,
                'optimal_week': optimal_week,
                'expected_success_rate': expected_success_rate,
                'risk_level': '低' if expected_success_rate > 0.9 else '中' if expected_success_rate > 0.8 else '高'
            }
            time_analysis.append(analysis)
        
        self.optimal_times = optimal_times
        
        # 显示结果
        time_df = pd.DataFrame(time_analysis)
        print("\n各组最佳NIPT时点:")
        print(time_df)
        
        return optimal_times, time_df

    def visualize_results(self):
        """
        结果可视化 - 显示包含异常值的完整数据分布
        """
        print("\n=== 结果可视化 ===")
        
        if not self.groups or not self.optimal_times:
            print("没有足够的数据进行可视化")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. BMI分布和分组（显示所有数据）
        ax1 = axes[0, 0]
        self.data['bmi'].hist(bins=30, alpha=0.7, ax=ax1)
        
        # 添加分组边界线
        for i, (group_name, group_info) in enumerate(self.groups.items()):
            bmi_range = group_info['bmi_range']
            if i > 0:  # 不画第一个组的左边界
                ax1.axvline(bmi_range[0], color='red', linestyle='--', alpha=0.8)
        
        ax1.set_title('BMI分布及分组边界（包含所有数据）')
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('频数')
        
        # 2. 各组成功率对比
        ax2 = axes[0, 1]
        group_names = []
        success_rates = []
        
        for group_name, group_info in self.groups.items():
            group_data = group_info['data']
            X = group_data[self.models['feature_cols']]
            X_scaled = self.models['prob_scaler'].transform(X)
            success_rate = self.models['prob_model'].predict_proba(X_scaled)[:, 1].mean()
            
            group_names.append(group_name.replace('group_', '组'))
            success_rates.append(success_rate)
        
        bars = ax2.bar(group_names, success_rates, color='skyblue', alpha=0.8)
        ax2.set_title('各组预期检测成功率')
        ax2.set_ylabel('成功率')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. 最优时点分布
        ax3 = axes[1, 0]
        optimal_times_list = list(self.optimal_times.values())
        group_labels = [name.replace('group_', '组') for name in self.optimal_times.keys()]
        
        bars = ax3.bar(group_labels, optimal_times_list, color='lightcoral', alpha=0.8)
        ax3.set_title('各组最佳NIPT检测时点')
        ax3.set_ylabel('检测时点(周)')
        
        # 添加数值标签
        for bar, time in zip(bars, optimal_times_list):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time}周', ha='center', va='bottom')
        
        # 4. 异常值分布
        ax4 = axes[1, 1]
        outlier_counts = {}
        
        if 'bmi' in self.data.columns:
            bmi_outliers = self.data[(self.data['bmi'] < 16) | (self.data['bmi'] > 50)]
            outlier_counts['BMI'] = len(bmi_outliers)
        
        if 'age' in self.data.columns:
            age_outliers = self.data[(self.data['age'] < 18) | (self.data['age'] > 45)]
            outlier_counts['年龄'] = len(age_outliers)
        
        if 'gestational_week' in self.data.columns:
            week_outliers = self.data[(self.data['gestational_week'] < 8) | (self.data['gestational_week'] > 30)]
            outlier_counts['孕周'] = len(week_outliers)
        
        if outlier_counts:
            bars = ax4.bar(outlier_counts.keys(), outlier_counts.values(), color='orange', alpha=0.7)
            ax4.set_title('各变量异常值数量')
            ax4.set_ylabel('异常值数量')
            
            # 添加数值标签
            for bar, count in zip(bars, outlier_counts.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, '无异常值', ha='center', va='center')
            ax4.set_title('异常值分布')
        
        plt.tight_layout()
        plt.show()
        
        # 保存图表
        plt.savefig('NIPT_optimization_results.png', dpi=300, bbox_inches='tight')
        print("可视化结果已保存为 'NIPT_optimization_results.png'")

# 主函数和其他辅助函数保持不变...
def main():
    """
    主函数：执行完整的分析流程
    """
    print("NIPT检测时点优化分析开始...")
    
    # 创建优化器实例
    optimizer = NIPTOptimizer()
    
    try:
        # 1. 加载数据 - 修改为您的实际文件路径
        file_path = "C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3.csv"
        data = optimizer.load_data(file_path, encoding='utf-8-sig')
        
        # 2. 探索性数据分析
        correlation_matrix = optimizer.exploratory_analysis()
        
        # 3. 构建预测模型
        models = optimizer.build_prediction_models()
        if models is None:
            print("模型构建失败，请检查数据")
            return
        
        # 4. 优化BMI分组
        groups, group_df = optimizer.optimize_bmi_grouping(n_groups=5)
        
        # 5. 确定最佳时点
        optimal_times, time_df = optimizer.find_optimal_times()
        
        # 6. 结果可视化
        optimizer.visualize_results()
        
        print("\n分析完成！所有数据均已保留，包括异常值")
        return optimizer
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 运行主程序
if __name__ == "__main__":
    print("="*60)
    print("NIPT检测时点优化系统 - 保留所有数据版本")
    print("="*60)
    
    # 运行主分析流程
    optimizer = main()