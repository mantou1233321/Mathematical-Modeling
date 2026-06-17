import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import multiprocessing
import os
warnings.filterwarnings('ignore')

class FixedNIPTOptimizer:
    """修复版 NIPT 优化器 - 最小改动修复关键问题"""

    def __init__(self, lambda_weight=0.8):
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

        # 筛选男胎数据
        self.male_data = self.data[
            (self.data['Y染色体浓度'].notna()) &
            (self.data['检测孕周'] >= 10) &
            (self.data['检测孕周'] <= 25) &
            (self.data['孕妇BMI'].notna()) &
            (self.data['身高'].notna()) &
            (self.data['体重'].notna()) &
            (self.data['年龄'].notna())
        ].copy()
        print(f"有效男胎数据：{len(self.male_data)}条")

    def analyze_bmi_groups(self):
        """分析 BMI 分组 - 如果没有聚类标签则创建"""
        print("\n=== BMI 分组分析 ===")

        if 'BMI_cluster' not in self.male_data.columns:
            print("未发现 BMI 聚类标签，基于 BMI 值创建分组...")
            bmi_values = self.male_data['孕妇BMI']
            quartiles = bmi_values.quantile([0.25, 0.5, 0.75]).values

            def assign_cluster(bmi):
                if bmi <= quartiles[0]:
                    return 0
                elif bmi <= quartiles[1]:
                    return 1
                elif bmi <= quartiles[2]:
                    return 2
                else:
                    return 3

            self.male_data['BMI_cluster'] = self.male_data['孕妇BMI'].apply(assign_cluster)

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

            print(f"分组{cluster}: BMI[{bmis.min():.2f}, {bmis.max():.2f}], "
                  f"均值={bmis.mean():.2f}, 样本数={len(cluster_data)}")

    def build_regression_models(self):
        """建立回归模型"""
        print("\n=== 随机森林回归模型建立 ===")

        for cluster, group_info in self.bmi_groups.items():
            data = group_info['data']

            if len(data) < 10:
                print(f"分组{cluster}样本数不足，跳过建模")
                continue

            X = data[['检测孕周', '孕妇BMI', '身高', '体重', '年龄']].copy()
            y = data['Y染色体浓度'].copy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = model.score(X_test, y_test)

            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            sigma = np.std(y_test - y_pred_test)

            self.regression_models[cluster] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'sigma': sigma,
                'feature_importances': model.feature_importances_,
                'feature_names': ['检测孕周', '孕妇BMI', '身高', '体重', '年龄']
            }

            print(f"分组{cluster}:")
            print(f"  测试集R²: {r2:.4f}")
            print(f"  交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  测试集RMSE: {test_rmse:.6f}")

    def improved_delay_risk(self, t):
        """改进的延迟风险函数"""
        if t <= 12:
            return 0.0
        elif t <= 16:
            return 0.1 * (t - 12) / 4
        elif t <= 20:
            return 0.1 + 0.3 * (t - 16) / 4
        elif t <= 24:
            return 0.4 + 0.4 * (t - 20) / 4
        else:
            return 0.8 + 0.2 * min((t - 24) / 2, 1.0)

    def enhanced_failure_risk(self, t, mean_features, model_params):
        """改进的失败风险函数"""
        model = model_params['model']
        feature_vector = np.array([
            t,
            mean_features['孕妇BMI'],
            mean_features['身高'],
            mean_features['体重'],
            mean_features['年龄']
        ]).reshape(1, -1)

        predicted_concentration = model.predict(feature_vector)[0]
        z_score = (0.04 - predicted_concentration) / model_params['sigma']
        failure_prob = stats.norm.cdf(z_score)

        return failure_prob, predicted_concentration

    def calculate_success_rate(self, t, group_data, model_params):
        """计算达标比例"""
        model = model_params['model']
        success_count = 0
        total_count = len(group_data)

        for _, patient in group_data.iterrows():
            feature_vector = np.array([
                t,
                patient['孕妇BMI'],
                patient['身高'],
                patient['体重'],
                patient['年龄']
            ]).reshape(1, -1)

            predicted_conc = model.predict(feature_vector)[0]
            if predicted_conc >= 0.04:
                success_count += 1

        return success_count / total_count

    def total_risk_enhanced(self, t, group_data, mean_features, model_params):
        """增强的总风险函数"""
        delay_r = self.improved_delay_risk(t)
        failure_r, pred_conc = self.enhanced_failure_risk(t, mean_features, model_params)
        success_rate = self.calculate_success_rate(t, group_data, model_params)

        total_risk = (
            self.lambda_weight * delay_r +
            (1 - self.lambda_weight) * failure_r -
            0.1 * success_rate
        )

        return total_risk, pred_conc, success_rate

    def find_optimal_timing(self):
        """寻找最优检测时点"""
        print(f"\n=== 最优时点计算 (λ={self.lambda_weight}) ===")

        for cluster, model_params in self.regression_models.items():
            group_data = self.bmi_groups[cluster]['data']
            mean_features = {
                '孕妇BMI': group_data['孕妇BMI'].mean(),
                '身高': group_data['身高'].mean(),
                '体重': group_data['体重'].mean(),
                '年龄': group_data['年龄'].mean()
            }

            def objective(t):
                risk, _, _ = self.total_risk_enhanced(t, group_data, mean_features, model_params)
                return risk

            result = minimize_scalar(objective, bounds=(10, 25), method='bounded')
            optimal_time = result.x
            min_risk = result.fun

            delay_r = self.improved_delay_risk(optimal_time)
            failure_r, predicted_conc = self.enhanced_failure_risk(optimal_time, mean_features, model_params)
            success_rate = self.calculate_success_rate(optimal_time, group_data, model_params)

            self.optimal_times[cluster] = {
                'optimal_time': optimal_time,
                'total_risk': min_risk,
                'delay_risk': delay_r,
                'failure_risk': failure_r,
                'predicted_concentration': predicted_conc,
                'success_rate': success_rate,
                'mean_features': mean_features
            }

            bmi_range = self.bmi_groups[cluster]
            print(f"\nBMI分组{cluster} ({bmi_range['min_bmi']:.2f}-{bmi_range['max_bmi']:.2f}):")
            print(f"  最优检测时点: {optimal_time:.2f}周")
            print(f"  总风险: {min_risk:.4f}")
            print(f"  - 延迟风险: {delay_r:.4f}")
            print(f"  - 失败风险: {failure_r:.4f}")
            print(f"  预期Y染色体浓度: {predicted_conc*100:.2f}%")
            print(f"  群体达标率: {success_rate*100:.1f}%")

    def generate_report(self, save_path=None):
        """生成报告"""
        report = []
        report.append("=" * 60)
        report.append("NIPT最优时点确定 - 修复版解答报告")
        report.append("=" * 60)

        report.append(f"\n数据概况:")
        report.append(f"总数据量: {len(self.data)}条")
        report.append(f"有效男胎数据: {len(self.male_data)}条")
        report.append(f"权重参数λ: {self.lambda_weight}")

        report.append(f"\nBMI分组结果:")
        for cluster, group_info in self.bmi_groups.items():
            report.append(f"分组{cluster}: BMI[{group_info['min_bmi']:.2f}, {group_info['max_bmi']:.2f}], "
                          f"均值={group_info['mean_bmi']:.2f}, 样本数={group_info['count']}")

        report.append(f"\n模型性能:")
        for cluster, model_params in self.regression_models.items():
            report.append(f"分组{cluster}: R²={model_params['r2']:.4f}, "
                          f"交叉验证R²={model_params['cv_r2_mean']:.4f}±{model_params['cv_r2_std']:.4f}")

        report.append(f"\n最优NIPT检测时点:")
        for cluster, optimal_info in self.optimal_times.items():
            bmi_range = self.bmi_groups[cluster]
            report.append(f"BMI分组{cluster} ({bmi_range['min_bmi']:.2f}-{bmi_range['max_bmi']:.2f}):")
            report.append(f"  最优检测时点: {optimal_info['optimal_time']:.2f}周")
            report.append(f"  预期Y染色体浓度: {optimal_info['predicted_concentration']*100:.2f}%")
            report.append(f"  群体达标率: {optimal_info['success_rate']*100:.1f}%")
            report.append(f"  综合风险: {optimal_info['total_risk']:.4f}")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")

        print(report_text)
        return report_text


def main_fixed():
    """修复版主函数"""

    analyzer = FixedNIPTOptimizer(lambda_weight=0.8)

    # 1. 加载数据
    data_path = "处理后的数据_带聚类.csv"  # 请修改为实际路径
    analyzer.load_data(data_path)

    # 2. BMI分组分析
    analyzer.analyze_bmi_groups()

    # 3. 建立回归模型
    analyzer.build_regression_models()

    # 4. 寻找最优时点
    analyzer.find_optimal_timing()

    # 5. 生成报告
    analyzer.generate_report('fixed_NIPT_report.txt')

    print("\n修复版分析完成！")


if __name__ == "__main__":
    # 2. 获取CPU核心数（确保进程数与核心数匹配，避免进程过多导致调度开销）
    cpu_cores = os.cpu_count() or 4  # 若无法获取则默认4核
    print(f"CPU核心数：{cpu_cores}，启动对应数量进程...")

    # 3. 创建进程池并执行任务（每个核心分配1个进程）
    pool = multiprocessing.Pool(processes=cpu_cores)
    # 向进程池提交任务（提交cpu_cores个任务，每个任务对应1个进程）
    for _ in range(cpu_cores):
        pool.apply_async(main_fixed())

    # 4. 阻塞进程池，防止主程序退出（任务会一直运行，需手动终止）
    pool.close()
    pool.join()
   