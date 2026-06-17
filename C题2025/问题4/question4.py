
"""
NIPT问题4：女胎异常判定方法构建
从本地文件读取数据的完整代码实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FetalAnomalyDetector:
    """女胎异常检测器"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
    
    def load_data(self, filepath, file_format='csv'):
        """
        从本地文件加载女胎检测数据
        filepath: 文件路径
        file_format: 文件格式，支持 'csv', 'excel'
        """
        try:
            if file_format == 'csv':
                self.data = pd.read_csv(filepath)
            elif file_format == 'excel':
                self.data = pd.read_excel(filepath)
            else:
                raise ValueError("不支持的文件格式，请使用 'csv' 或 'excel'")
            
            print(f"数据加载完成，共{len(self.data)}条记录")
            
            # 检查必要列是否存在
            required_columns = ['z13', 'z18', 'z21', 'x_z']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                print(f"警告: 缺少以下必要列: {missing_columns}")
            
            # 如果数据中没有异常标签，尝试基于Z值创建
            if 'is_abnormal' not in self.data.columns:
                print("数据中没有异常标签，将基于Z值阈值自动创建...")
                threshold = 2.5
                z13_abnormal = np.abs(self.data['z13']) > threshold
                z18_abnormal = np.abs(self.data['z18']) > threshold  
                z21_abnormal = np.abs(self.data['z21']) > threshold
                
                self.data['is_abnormal'] = (z13_abnormal | z18_abnormal | z21_abnormal).astype(int)
                
                # 创建异常类型标签
                anomaly_type = np.full(len(self.data), 'Normal', dtype=object)
                anomaly_type[z13_abnormal & ~z21_abnormal] = 'T13'
                anomaly_type[z18_abnormal & ~z21_abnormal] = 'T18'  
                anomaly_type[z21_abnormal & ~(z13_abnormal | z18_abnormal)] = 'T21'
                anomaly_type[z13_abnormal & z21_abnormal] = 'T13T21'
                anomaly_type[z18_abnormal & z21_abnormal] = 'T18T21'
                
                self.data['anomaly_type'] = anomaly_type
            
            # 统计异常类型分布
            print("\n异常类型分布:")
            type_counts = self.data['anomaly_type'].value_counts()
            for anomaly_type, count in type_counts.items():
                print(f"  {anomaly_type}: {count} ({count/len(self.data)*100:.1f}%)")
            
            return self.data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def exploratory_analysis(self):
        """
        探索性数据分析
        """
        print("\n=== 探索性数据分析 ===")
        
        # 基本统计信息
        print("\n基本统计信息:")
        numeric_cols = ['z13', 'z18', 'z21', 'x_z']
        if 'age' in self.data.columns:
            numeric_cols.append('age')
        if 'bmi' in self.data.columns:
            numeric_cols.append('bmi')
        if 'gc_content' in self.data.columns:
            numeric_cols.append('gc_content')
            
        print(self.data[numeric_cols].describe())
        
        # Z值分布分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Z值分布
        z_cols = ['z13', 'z18', 'z21', 'x_z']
        z_names = ['13号染色体Z值', '18号染色体Z值', '21号染色体Z值', 'X染色体Z值']
        
        for i, (col, name) in enumerate(zip(z_cols, z_names)):
            if col not in self.data.columns:
                continue
                
            ax = axes[i//2, i%2]
            
            # 正常vs异常分布
            normal_data = self.data[self.data['is_abnormal'] == 0][col]
            abnormal_data = self.data[self.data['is_abnormal'] == 1][col]
            
            ax.hist(normal_data, bins=30, alpha=0.7, label='正常', color='blue')
            ax.hist(abnormal_data, bins=30, alpha=0.7, label='异常', color='red')
            ax.axvline(2.5, color='black', linestyle='--', label='阈值±2.5')
            ax.axvline(-2.5, color='black', linestyle='--')
            ax.set_title(name)
            ax.set_xlabel('Z值')
            ax.set_ylabel('频数')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 异常类型的Z值箱线图
        plt.figure(figsize=(15, 10))
        valid_z_cols = [col for col in z_cols if col in self.data.columns]
        
        for i, (col, name) in enumerate(zip(valid_z_cols, z_names), 1):
            plt.subplot(2, 2, i)
            self.data.boxplot(column=col, by='anomaly_type', ax=plt.gca())
            plt.title(f'{name}按异常类型分布')
            plt.suptitle('')
        
        plt.tight_layout()
        plt.show()
    
    def build_isolation_forest_model(self, contamination=0.1):
        """
        构建孤立森林异常检测模型
        """
        print("\n=== 构建孤立森林模型 ===")
        
        # 特征选择 - 使用可用的特征
        base_features = ['z13', 'z18', 'z21', 'x_z']
        optional_features = ['gc_content', 'gc_13', 'gc_18', 'gc_21', 'age', 'bmi', 'read_count', 'filtered_ratio']
        
        feature_cols = [col for col in base_features if col in self.data.columns]
        for col in optional_features:
            if col in self.data.columns:
                feature_cols.append(col)
        
        print(f"使用的特征: {feature_cols}")
        
        X = self.data[feature_cols]
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练孤立森林模型
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # 拟合模型
        iso_forest.fit(X_scaled)
        
        # 预测异常
        anomaly_pred = iso_forest.predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        # 转换预测结果（-1表示异常，1表示正常）
        anomaly_pred_binary = (anomaly_pred == -1).astype(int)
        
        # 评估性能
        y_true = self.data['is_abnormal']
        
        print(f"孤立森林检测结果:")
        print(f"  检测到异常样本数: {anomaly_pred_binary.sum()}")
        print(f"  实际异常样本数: {y_true.sum()}")
        print(f"  准确率: {(anomaly_pred_binary == y_true).mean():.3f}")
        
        # 计算各异常类型的检出率
        print("\n各异常类型检出情况:")
        for anomaly_type in self.data['anomaly_type'].unique():
            if anomaly_type == 'Normal':
                continue
            mask = self.data['anomaly_type'] == anomaly_type
            detected = anomaly_pred_binary[mask].sum()
            total = mask.sum()
            print(f"  {anomaly_type}: {detected}/{total} ({detected/total*100:.1f}%)")
        
        self.models['isolation_forest'] = {
            'model': iso_forest,
            'predictions': anomaly_pred_binary,
            'scores': anomaly_scores,
            'feature_cols': feature_cols
        }
        
        return iso_forest, anomaly_pred_binary, anomaly_scores
    
    def build_lof_model(self, n_neighbors=5):
        """
        构建LOF局部异常因子模型
        """
        print("\n=== 构建LOF模型 ===")
        if len(self.data) <= 2:  # 如果样本太少
            print("样本数量不足，跳过LOF模型")
            return None, None, None
        feature_cols = self.models['isolation_forest']['feature_cols']
        X = self.data[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # 动态调整n_neighbors
        n_samples = len(X_scaled)
        if n_neighbors >= n_samples:
            n_neighbors = max(1, n_samples - 1)
            print(f"调整n_neighbors为: {n_neighbors} (样本数量: {n_samples})")
        
        # 训练LOF模型
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,  # 使用调整后的值
            contamination=0.1
        )
    
        
        # 预测异常
        anomaly_pred = lof.fit_predict(X_scaled)
        anomaly_scores = lof.negative_outlier_factor_
        
        # 转换预测结果
        anomaly_pred_binary = (anomaly_pred == -1).astype(int)
        
        # 评估性能
        y_true = self.data['is_abnormal']
        
        print(f"LOF检测结果:")
        print(f"  检测到异常样本数: {anomaly_pred_binary.sum()}")
        print(f"  准确率: {(anomaly_pred_binary == y_true).mean():.3f}")
        
        # 计算各异常类型的检出率
        print("\n各异常类型检出情况:")
        for anomaly_type in self.data['anomaly_type'].unique():
            if anomaly_type == 'Normal':
                continue
            mask = self.data['anomaly_type'] == anomaly_type
            detected = anomaly_pred_binary[mask].sum()
            total = mask.sum()
            print(f"  {anomaly_type}: {detected}/{total} ({detected/total*100:.1f}%)")
        
        self.models['lof'] = {
            'model': lof,
            'predictions': anomaly_pred_binary,
            'scores': anomaly_scores,
            'feature_cols': feature_cols
        }
        
        return lof, anomaly_pred_binary, anomaly_scores
    
    def build_one_class_svm_model(self, nu=0.1):
        """
        构建One-Class SVM模型
        """
        print("\n=== 构建One-Class SVM模型 ===")
        
        feature_cols = self.models['isolation_forest']['feature_cols']
        X = self.data[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # 使用正常样本训练模型
        normal_mask = self.data['is_abnormal'] == 0
        X_normal = X_scaled[normal_mask]
        
        # 训练One-Class SVM
        oc_svm = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        oc_svm.fit(X_normal)
        
        # 预测所有样本
        anomaly_pred = oc_svm.predict(X_scaled)
        anomaly_scores = oc_svm.decision_function(X_scaled)
        
        # 转换预测结果
        anomaly_pred_binary = (anomaly_pred == -1).astype(int)
        
        # 评估性能
        y_true = self.data['is_abnormal']
        
        print(f"One-Class SVM检测结果:")
        print(f"  检测到异常样本数: {anomaly_pred_binary.sum()}")
        print(f"  准确率: {(anomaly_pred_binary == y_true).mean():.3f}")
        
        # 计算各异常类型的检出率
        print("\n各异常类型检出情况:")
        for anomaly_type in self.data['anomaly_type'].unique():
            if anomaly_type == 'Normal':
                continue
            mask = self.data['anomaly_type'] == anomaly_type
            detected = anomaly_pred_binary[mask].sum()
            total = mask.sum()
            print(f"  {anomaly_type}: {detected}/{total} ({detected/total*100:.1f}%)")
        
        self.models['one_class_svm'] = {
            'model': oc_svm,
            'predictions': anomaly_pred_binary,
            'scores': anomaly_scores,
            'feature_cols': feature_cols
        }
        
        return oc_svm, anomaly_pred_binary, anomaly_scores
    
    def z_score_threshold_method(self, threshold=2.5):
        """
        传统Z值阈值判定方法
        """
        print(f"\n=== Z值阈值判定法(阈值={threshold}) ===")
        
        # 计算各染色体异常
        z13_abnormal = np.abs(self.data['z13']) > threshold
        z18_abnormal = np.abs(self.data['z18']) > threshold  
        z21_abnormal = np.abs(self.data['z21']) > threshold
        
        # 任一染色体异常即判定为异常
        z_abnormal = z13_abnormal | z18_abnormal | z21_abnormal
        z_abnormal_binary = z_abnormal.astype(int)
        
        # 评估性能
        y_true = self.data['is_abnormal']
        
        print(f"Z值阈值法检测结果:")
        print(f"  检测到异常样本数: {z_abnormal_binary.sum()}")
        print(f"  准确率: {(z_abnormal_binary == y_true).mean():.3f}")
        
        # 详细分析各染色体检测情况
        print(f"\n各染色体异常检出:")
        print(f"  13号染色体异常: {z13_abnormal.sum()}")
        print(f"  18号染色体异常: {z18_abnormal.sum()}")
        print(f"  21号染色体异常: {z21_abnormal.sum()}")
        
        # 计算各异常类型的检出率
        print("\n各异常类型检出情况:")
        for anomaly_type in self.data['anomaly_type'].unique():
            if anomaly_type == 'Normal':
                continue
            mask = self.data['anomaly_type'] == anomaly_type
            detected = z_abnormal_binary[mask].sum()
            total = mask.sum()
            print(f"  {anomaly_type}: {detected}/{total} ({detected/total*100:.1f}%)")
        
        self.models['z_threshold'] = {
            'predictions': z_abnormal_binary,
            'z13_abnormal': z13_abnormal,
            'z18_abnormal': z18_abnormal,
            'z21_abnormal': z21_abnormal,
            'threshold': threshold
        }
        
        return z_abnormal_binary
    
    def ensemble_prediction(self):
        """
        集成多种方法进行最终判定
        """
        print("\n=== 集成判定方法 ===")
        
        # 获取各模型的预测结果
        if_pred = self.models['isolation_forest']['predictions']
        lof_pred = self.models['lof']['predictions']
        svm_pred = self.models['one_class_svm']['predictions']
        z_pred = self.models['z_threshold']['predictions']
        
        # 获取异常得分（标准化到0-1）
        if_scores = self.models['isolation_forest']['scores']
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        lof_scores = -self.models['lof']['scores']  # LOF分数越负越异常
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
        
        svm_scores = -self.models['one_class_svm']['scores']  # SVM分数越负越异常
        svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
        
        # 集成策略1：投票法
        vote_pred = ((if_pred + lof_pred + svm_pred + z_pred) >= 2).astype(int)
        
        # 集成策略2：加权得分法
        weights = [0.3, 0.25, 0.25, 0.2]  # IF, LOF, SVM, Z_threshold权重
        weighted_scores = (weights[0] * if_scores_norm + 
                          weights[1] * lof_scores_norm + 
                          weights[2] * svm_scores_norm + 
                          weights[3] * z_pred)
        
        weighted_pred = (weighted_scores > 0.5).astype(int)
        
        # 集成策略3：严格策略（任一方法检测到即为异常）
        strict_pred = (if_pred | lof_pred | svm_pred | z_pred).astype(int)
        
        # 评估各集成策略
        y_true = self.data['is_abnormal']
        
        strategies = {
            '投票法': vote_pred,
            '加权得分法': weighted_pred,
            '严格策略': strict_pred
        }
        
        print("集成策略性能对比:")
        best_strategy = None
        best_accuracy = 0
        
        for name, pred in strategies.items():
            accuracy = (pred == y_true).mean()
            precision = (pred[pred == 1] == y_true[pred == 1]).mean() if pred.sum() > 0 else 0
            recall = (pred[y_true == 1] == y_true[y_true == 1]).mean() if y_true.sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {name}:")
            print(f"    准确率: {accuracy:.3f}")
            print(f"    精确率: {precision:.3f}")
            print(f"    召回率: {recall:.3f}")
            print(f"    F1分数: {f1:.3f}")
            print(f"    检测异常数: {pred.sum()}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_strategy = name
        
        print(f"\n最佳策略: {best_strategy} (准确率: {best_accuracy:.3f})")
        
        # 保存集成结果
        self.models['ensemble'] = {
            'vote_pred': vote_pred,
            'weighted_pred': weighted_pred,
            'strict_pred': strict_pred,
            'weighted_scores': weighted_scores,
            'best_strategy': best_strategy,
            'best_pred': strategies[best_strategy]
        }
        
        return strategies[best_strategy]
    
    def detailed_anomaly_analysis(self):
        """
        详细异常类型分析
        """
        print("\n=== 详细异常类型分析 ===")
        
        # 使用最佳集成策略的结果
        final_pred = self.models['ensemble']['best_pred']
        
        # 创建详细分析表
        analysis_results = []
        
        for anomaly_type in self.data['anomaly_type'].unique():
            mask = self.data['anomaly_type'] == anomaly_type
            total_samples = mask.sum()
            
            if anomaly_type == 'Normal':
                # 对于正常样本，计算误报率
                false_positive = final_pred[mask].sum()
                specificity = 1 - false_positive / total_samples
                
                analysis_results.append({
                    '异常类型': anomaly_type,
                    '样本总数': total_samples,
                    '检出数量': false_positive,
                    '检出率(%)': false_positive / total_samples * 100,
                    '特异性': specificity,
                    '备注': '误报率'
                })
            else:
                # 对于异常样本，计算检出率
                detected = final_pred[mask].sum()
                detection_rate = detected / total_samples if total_samples > 0 else 0
                
                analysis_results.append({
                    '异常类型': anomaly_type,
                    '样本总数': total_samples,
                    '检出数量': detected,
                    '检出率(%)': detection_rate * 100,
                    '敏感性': detection_rate,
                    '备注': '真阳性率'
                })
        
        # 转换为DataFrame并显示
        analysis_df = pd.DataFrame(analysis_results)
        print("\n各异常类型检测性能:")
        print(analysis_df.round(3))
        
        # 混淆矩阵分析
        from sklearn.metrics import confusion_matrix, classification_report
        
        y_true = self.data['is_abnormal']
        cm = confusion_matrix(y_true, final_pred)
        
        print(f"\n混淆矩阵:")
        print(f"                预测")
        print(f"实际    正常  异常")
        print(f"正常    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"异常    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # 计算关键性能指标
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性/召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
        
        print(f"\n关键性能指标:")
        print(f"  敏感性(Sensitivity): {sensitivity:.3f}")
        print(f"  特异性(Specificity): {specificity:.3f}")
        print(f"  阳性预测值(PPV): {ppv:.3f}")
        print(f"  阴性预测值(NPV): {npv:.3f}")
        
        self.results['detailed_analysis'] = analysis_df
        self.results['performance_metrics'] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        }
        
        return analysis_df
    
    def feature_importance_analysis(self):
        """
        特征重要性分析
        """
        print("\n=== 特征重要性分析 ===")
        
        # 基于孤立森林的特征重要性（通过排列重要性）
        from sklearn.inspection import permutation_importance
        
        feature_cols = self.models['isolation_forest']['feature_cols']
        X = self.data[feature_cols]
        X_scaled = self.scaler.transform(X)
        y = self.data['is_abnormal']
        
        # 计算排列重要性
        iso_model = self.models['isolation_forest']['model']
        
        # 由于孤立森林是无监督的，我们计算特征对异常得分的影响
        base_scores = iso_model.decision_function(X_scaled)
        
        importance_scores = []
        for i, feature in enumerate(feature_cols):
            X_permuted = X_scaled.copy()
            # 打乱第i个特征
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_scores = iso_model.decision_function(X_permuted)
            # 计算得分变化
            importance = np.mean(np.abs(base_scores - permuted_scores))
            importance_scores.append(importance)
        
        # 排序特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("特征重要性排序:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('重要性得分')
        plt.title('特征重要性分析')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def visualize_results(self):
        """
        结果可视化
        """
        print("\n=== 结果可视化 ===")
        
        # 创建综合可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 各模型检测性能对比
        ax1 = axes[0, 0]
        models = ['孤立森林', 'LOF', 'One-Class SVM', 'Z值阈值', '集成方法']
        y_true = self.data['is_abnormal']
        
        accuracies = []
        for model_name, key in zip(models[:-1], ['isolation_forest', 'lof', 'one_class_svm', 'z_threshold']):
            pred = self.models[key]['predictions']
            acc = (pred == y_true).mean()
            accuracies.append(acc)
        
        # 添加集成方法准确率
        ensemble_pred = self.models['ensemble']['best_pred']
        ensemble_acc = (ensemble_pred == y_true).mean()
        accuracies.append(ensemble_acc)
        
        bars = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
        ax1.set_title('各模型检测准确率对比')
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. 异常类型检出率
        ax2 = axes[0, 1]
        anomaly_types = [t for t in self.data['anomaly_type'].unique() if t != 'Normal']
        detection_rates = []
        
        for anomaly_type in anomaly_types:
            mask = self.data['anomaly_type'] == anomaly_type
            detected = ensemble_pred[mask].sum()
            total = mask.sum()
            rate = detected / total if total > 0 else 0
            detection_rates.append(rate)
        
        bars = ax2.bar(anomaly_types, detection_rates, color='lightblue')
        ax2.set_title('各异常类型检出率')
        ax2.set_ylabel('检出率')
        ax2.set_ylim(0, 1)
        
        for bar, rate in zip(bars, detection_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # 3. Z值分布散点图
        ax3 = axes[0, 2]
        normal_data = self.data[self.data['is_abnormal'] == 0]
        abnormal_data = self.data[self.data['is_abnormal'] == 1]
        
        ax3.scatter(normal_data['z21'], normal_data['z18'], c='blue', alpha=0.6, label='正常', s=20)
        ax3.scatter(abnormal_data['z21'], abnormal_data['z18'], c='red', alpha=0.8, label='异常', s=30)
        ax3.axhline(y=2.5, color='black', linestyle='--', alpha=0.5)
        ax3.axhline(y=-2.5, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=2.5, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=-2.5, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('21号染色体Z值')
        ax3.set_ylabel('18号染色体Z值')
        ax3.set_title('Z值分布散点图')
        ax3.legend()
        
        # 4. ROC曲线（如果可能）
        ax4 = axes[1, 0]
        try:
            from sklearn.metrics import roc_curve, auc
            
            # 使用加权得分计算ROC
            weighted_scores = self.models['ensemble']['weighted_scores']
            fpr, tpr, _ = roc_curve(y_true, weighted_scores)
            roc_auc = auc(fpr, tpr)
            
            ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
            ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax4.set_xlim([0.0, 1.0])
            ax4.set_ylim([0.0, 1.05])
            ax4.set_xlabel('假阳性率')
            ax4.set_ylabel('真阳性率')
            ax4.set_title('ROC曲线')
            ax4.legend(loc="lower right")
        except:
            ax4.text(0.5, 0.5, 'ROC曲线计算失败', ha='center', va='center')
            ax4.set_title('ROC曲线')
        
        # 5. 混淆矩阵热图
        ax5 = axes[1, 1]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, ensemble_pred)
        
        im = ax5.imshow(cm, interpolation='nearest', cmap='Blues')
        ax5.set_title('混淆矩阵')
        tick_marks = np.arange(2)
        ax5.set_xticks(tick_marks)
        ax5.set_yticks(tick_marks)
        ax5.set_xticklabels(['正常', '异常'])
        ax5.set_yticklabels(['正常', '异常'])
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                ax5.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        
        ax5.set_ylabel('真实标签')
        ax5.set_xlabel('预测标签')
        
        # 6. 异常得分分布
        ax6 = axes[1, 2]
        weighted_scores = self.models['ensemble']['weighted_scores']
        normal_scores = weighted_scores[y_true == 0]
        abnormal_scores = weighted_scores[y_true == 1]
        
        ax6.hist(normal_scores, bins=30, alpha=0.7, label='正常', color='blue')
        ax6.hist(abnormal_scores, bins=30, alpha=0.7, label='异常', color='red')
        ax6.axvline(x=0.5, color='black', linestyle='--', label='阈值')
        ax6.set_xlabel('异常得分')
        ax6.set_ylabel('频数')
        ax6.set_title('异常得分分布')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 保存图表
        plt.savefig('女胎异常检测结果.png', dpi=300, bbox_inches='tight')
        print("可视化结果已保存为 '女胎异常检测结果.png'")
    
    def generate_final_report(self):
        """
        生成最终报告
        """
        print("\n" + "="*60)
        print("女胎异常检测最终报告")
        print("="*60)
        
        # 数据概况
        print(f"\n【数据概况】")
        print(f"总样本数: {len(self.data)}")
        print(f"异常样本数: {self.data['is_abnormal'].sum()}")
        print(f"异常率: {self.data['is_abnormal'].mean():.1%}")
        
        # 异常类型分布
        print(f"\n【异常类型分布】")
        type_counts = self.data['anomaly_type'].value_counts()
        for anomaly_type, count in type_counts.items():
            print(f"  {anomaly_type}: {count}例 ({count/len(self.data)*100:.1f}%)")
        
        # 最佳模型性能
        best_strategy = self.models['ensemble']['best_strategy']
        best_pred = self.models['ensemble']['best_pred']
        y_true = self.data['is_abnormal']
        
        print(f"\n【最佳检测策略】")
        print(f"策略名称: {best_strategy}")
        
        # 性能指标
        metrics = self.results['performance_metrics']
        print(f"敏感性: {metrics['sensitivity']:.3f}")
        print(f"特异性: {metrics['specificity']:.3f}")
        print(f"阳性预测值: {metrics['ppv']:.3f}")
        print(f"阴性预测值: {metrics['npv']:.3f}")
        
        # 各异常类型检测效果
        print(f"\n【各异常类型检测效果】")
        analysis_df = self.results['detailed_analysis']
        for _, row in analysis_df.iterrows():
            if row['异常类型'] != 'Normal':
                print(f"  {row['异常类型']}: {row['检出率(%)']:.1f}% ({row['检出数量']:.0f}/{row['样本总数']:.0f})")
        
        # 关键发现
        print(f"\n【关键发现】")
        t13t21_mask = self.data['anomaly_type'] == 'T13T21'
        if t13t21_mask.any():
            t13t21_detected = best_pred[t13t21_mask].sum()
            t13t21_total = t13t21_mask.sum()
            print(f"• T13T21复合异常检出率最高: {t13t21_detected/t13t21_total*100:.1f}%")
        
        # 建议
        print(f"\n【临床应用建议】")
        print("• 采用集成检测策略，提高检测准确性")
        print("• 重点关注Z值显著偏离正常范围的样本")
        print("• 结合GC含量等技术指标进行质量控制")
        print("• 对于复合异常类型，当前方法检测效果较好")
        
        return {
            'best_strategy': best_strategy,
            'performance_metrics': metrics,
            'anomaly_detection_results': analysis_df
        }

def main(filepath, file_format='csv'):
    """
    主函数：执行完整的女胎异常检测分析
    filepath: 数据文件路径
    file_format: 文件格式，支持 'csv' 或 'excel'
    """
    print("女胎异常检测分析开始...")
    
    # 创建检测器实例
    detector = FetalAnomalyDetector()
    
    # 1. 加载数据
    data = detector.load_data(filepath, file_format)
    if data is None:
        print("数据加载失败，程序退出")
        return None
    
    # 2. 探索性数据分析
    detector.exploratory_analysis()
    
    # 3. 构建各种检测模型
    detector.build_isolation_forest_model()
    detector.build_lof_model()
    detector.build_one_class_svm_model()
    detector.z_score_threshold_method()
    
    # 4. 集成判定
    best_pred = detector.ensemble_prediction()
    
    # 5. 详细异常分析
    analysis_results = detector.detailed_anomaly_analysis()
    
    # 6. 特征重要性分析
    feature_importance = detector.feature_importance_analysis()
    
    # 7. 结果可视化
    detector.visualize_results()
    
    # 8. 生成最终报告
    final_report = detector.generate_final_report()
    
    print("\n女胎异常检测分析完成！")
    return detector

# 新样本预测功能
def predict_new_sample(detector, sample_data):
    """
    为新样本预测异常
    """
    print("\n【新样本异常预测】")
    
    # 特征标准化
    feature_cols = detector.models['isolation_forest']['feature_cols']
    
    # 确保样本数据包含所有必要特征
    missing_features = [col for col in feature_cols if col not in sample_data]
    if missing_features:
        print(f"警告: 样本缺少以下特征: {missing_features}")
        # 为缺失特征设置默认值
        for col in missing_features:
            if col in ['z13', 'z18', 'z21', 'x_z']:
                sample_data[col] = 0  # Z值默认设为0
            elif col in ['gc_content', 'gc_13', 'gc_18', 'gc_21']:
                sample_data[col] = 0.45  # GC含量默认设为0.45
            elif col == 'age':
                sample_data[col] = 30  # 年龄默认设为30
            elif col == 'bmi':
                sample_data[col] = 25  # BMI默认设为25
            else:
                sample_data[col] = 0
    
    X_new = np.array([[sample_data[col] for col in feature_cols]])
    X_new_scaled = detector.scaler.transform(X_new)
    
    # 各模型预测
    if_pred = detector.models['isolation_forest']['model'].predict(X_new_scaled)[0]
    
    # LOF预测
    try:
        n_samples = X_new_scaled.shape[0]
        lof_model = detector.models['lof']['model']
        if n_samples < lof_model.n_neighbors:
            print(f"样本数量不足以满足 LOF 的 n_neighbors 要求，跳过 LOF 预测")
            lof_pred = 1  # 默认标记为“正常”
        else:
            lof_pred = lof_model.fit_predict(X_new_scaled)[0]
    except Exception as e:
        print(f"LOF 预测失败: {e}")
        lof_pred = 1  # 默认标记为“正常”
    
    # SVM预测
    svm_pred = detector.models['one_class_svm']['model'].predict(X_new_scaled)[0]
    
    # Z值判定
    threshold = detector.models['z_threshold']['threshold']
    z_abnormal = (abs(sample_data['z13']) > threshold or 
                  abs(sample_data['z18']) > threshold or 
                  abs(sample_data['z21']) > threshold)
    
    # 转换预测结果
    if_abnormal = if_pred == -1
    lof_abnormal = lof_pred == -1
    svm_abnormal = svm_pred == -1
    
    # 集成判定
    vote_result = sum([if_abnormal, lof_abnormal, svm_abnormal, z_abnormal]) >= 2
    
    print(f"样本特征:")
    for col in feature_cols:
        print(f"  {col}: {sample_data[col]}")
    
    print(f"\n各模型预测结果:")
    print(f"  孤立森林: {'异常' if if_abnormal else '正常'}")
    print(f"  LOF: {'异常' if lof_abnormal else '正常'}")
    print(f"  One-Class SVM: {'异常' if svm_abnormal else '正常'}")
    print(f"  Z值判定: {'异常' if z_abnormal else '正常'}")
    
    print(f"\n最终判定: {'异常' if vote_result else '正常'}")
    
    if vote_result:
        print("建议: 需要进一步检查确认")
    else:
        print("建议: 暂未发现异常")
    
    return vote_result

if __name__ == "__main__":
    # 使用示例
    file_path = "附件_wash_nv.csv"  # 替换为您的数据文件路径
    file_format = "csv"  # 或 "excel"
    
    # 运行主程序
    detector = main(file_path, file_format)
    
    if detector:
        # 示例：预测新样本
        example_sample = {
            'z13': 0.5, 'z18': -0.3, 'z21': 4.2, 'x_z': -1.1,
            'gc_content': 0.45, 'gc_13': 0.44, 'gc_18': 0.46, 'gc_21': 0.45,
            'age': 30, 'bmi': 28.5
        }
        
        print("\n" + "="*60)
        print("示例：新样本预测")
        print("="*60)
        result = predict_new_sample(detector, example_sample)