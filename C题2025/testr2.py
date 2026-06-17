"""
测试指定模型公式在NIPT数据上的R²值
模型公式: Y(标准化) = -0.59 +0.064×孕周 +0.021×BMI -0.0037×孕周² +0.0000692×孕周³ -0.0004×BMI²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def test_model_formula(file_path):
    """
    测试指定模型公式在数据上的表现
    
    Parameters:
    file_path: str, 数据文件路径
    """
    
    print("="*60)
    print("测试模型公式在NIPT数据上的R²值")
    print("="*60)
    
    try:
        # 读取数据
        print(f"正在从 {file_path} 读取数据...")
        data = pd.read_csv(file_path)
        
        # 检查必要的列
        required_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据中缺少必要的列: {missing_columns}")
        
        print(f"成功读取数据，共 {len(data)} 行")
        
        # 提取变量
        y_true = data['Y染色体浓度'].values
        week = data['检测孕周'].values
        bmi = data['孕妇BMI'].values
        
        # 定义模型公式
        def model_predict(week, bmi):
            """根据给定的模型公式进行预测"""
            return (-0.59 + 
                    0.064 * week + 
                    0.021 * bmi - 
                    0.0037 * (week ** 2) + 
                    0.0000692 * (week ** 3) - 
                    0.0004 * (bmi ** 2))
        
        # 计算预测值
        y_pred = model_predict(week, bmi)
        
        # 计算R²
        r2 = r2_score(y_true, y_pred)
        
        # 计算其他评估指标
        residuals = y_true - y_pred
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # 输出结果
        print("\n模型评估结果:")
        print(f"R² = {r2:.6f}")
        print(f"调整R² = {1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - 6):.6f}")
        print(f"MSE = {mse:.6f}")
        print(f"RMSE = {rmse:.6f}")
        print(f"MAE = {mae:.6f}")
        
        # 残差分析
        print(f"\n残差统计:")
        print(f"残差均值: {np.mean(residuals):.6f}")
        print(f"残差标准差: {np.std(residuals):.6f}")
        print(f"最小残差: {np.min(residuals):.6f}")
        print(f"最大残差: {np.max(residuals):.6f}")
        
        # 绘制预测值与真实值的散点图
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'预测值 vs 真实值 (R² = {r2:.4f})')
        plt.grid(True, alpha=0.3)
        
        # 绘制残差图
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 按孕周分组分析
        print(f"\n按孕周分组的模型表现:")
        week_bins = np.quantile(week, [0, 0.25, 0.5, 0.75, 1.0])
        week_labels = ['低孕周', '中低孕周', '中高孕周', '高孕周']
        
        for i in range(len(week_bins) - 1):
            mask = (week >= week_bins[i]) & (week < week_bins[i + 1])
            if np.sum(mask) > 0:
                group_r2 = r2_score(y_true[mask], y_pred[mask])
                print(f"{week_labels[i]}: {np.sum(mask)}个样本, R² = {group_r2:.4f}")
        
        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'y_true': y_true,
            'y_pred': y_pred,
            'residuals': residuals
        }
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保文件路径正确且文件存在")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def compare_with_linear_model(file_path):
    """
    与简单线性模型进行比较
    """
    print("\n" + "="*60)
    print("与简单线性模型比较")
    print("="*60)
    
    try:
        data = pd.read_csv(file_path)
        y_true = data['Y染色体浓度'].values
        week = data['检测孕周'].values
        bmi = data['孕妇BMI'].values
        
        # 简单线性模型 (仅孕周和BMI的线性项)
        X_linear = np.column_stack([np.ones(len(y_true)), week, bmi])
        beta_linear = np.linalg.lstsq(X_linear, y_true, rcond=None)[0]
        y_pred_linear = X_linear @ beta_linear
        r2_linear = r2_score(y_true, y_pred_linear)
        
        # 多项式模型 (使用给定的公式)
        y_pred_poly = (-0.59 + 0.064 * week + 0.021 * bmi - 
                      0.0037 * (week ** 2) + 0.0000692 * (week ** 3) - 
                      0.0004 * (bmi ** 2))
        r2_poly = r2_score(y_true, y_pred_poly)
        
        print(f"简单线性模型 R²: {r2_linear:.6f}")
        print(f"多项式模型 R²: {r2_poly:.6f}")
        print(f"改进: {(r2_poly - r2_linear):.6f} ({((r2_poly - r2_linear)/r2_linear*100):.2f}%)")
        
        return r2_linear, r2_poly
        
    except Exception as e:
        print(f"比较时发生错误: {str(e)}")
        return None, None

# 主程序
if __name__ == "__main__":
    # 文件路径 - 请修改为您的实际路径
    file_path = "C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3.csv"
    
    # 测试模型公式
    results = test_model_formula(file_path)
    
    if results is not None:
        # 与线性模型比较
        r2_linear, r2_poly = compare_with_linear_model(file_path)
        
        print("\n" + "="*60)
        print("最终总结")
        print("="*60)
        print(f"模型公式: Y = -0.59 + 0.064×孕周 + 0.021×BMI - 0.0037×孕周² + 0.0000692×孕周³ - 0.0004×BMI²")
        print(f"在数据集上的R²: {results['r2']:.6f}")
        
        if r2_linear is not None:
            print(f"相比简单线性模型的改进: {r2_poly - r2_linear:.6f}")
        
        # 保存预测结果
        output_df = pd.DataFrame({
            '真实值': results['y_true'],
            '预测值': results['y_pred'],
            '残差': results['residuals']
        })
        output_path = "model_predictions.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {output_path}")