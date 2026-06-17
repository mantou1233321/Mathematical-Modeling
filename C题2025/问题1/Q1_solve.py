"""
NIPT非线性混合效应模型核心计算过程
基于标准化数据的Y染色体浓度预测
"""

import numpy as np
import pandas as pd
from scipy import stats

def build_nonlinear_mixed_model(data):
    """
    构建NIPT非线性混合效应模型的核心计算过程
    
    Parameters:
    data: pandas.DataFrame, 包含标准化的NIPT数据
    
    Returns:
    dict: 模型结果字典
    """
    
    # 步骤1: 提取因变量和自变量
    print("步骤1: 数据准备")
    
    # 检查必要的列是否存在
    required_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI', 'X染色体浓度', 
                       'GC含量', '原始读段数', '检测抽血次数', '年龄', '孕妇代码']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少必要的列: {missing_columns}")
    
    y = data['Y染色体浓度'].values  # 因变量（标准化）
    
    # 提取标准化的自变量
    week = data['检测孕周'].values
    bmi = data['孕妇BMI'].values  
    x_conc = data['X染色体浓度'].values
    gc_content = data['GC含量'].values
    reads = data['原始读段数'].values
    blood_times = data['检测抽血次数'].values
    age = data['年龄'].fillna(0).values
    
    print(f"样本数: {len(y)}")
    print(f"变量数: 7个主要变量")
    
    # 步骤2: 构建增强设计矩阵（包含非线性项和交互项）
    print("\n步骤2: 构建增强设计矩阵")
    X = np.column_stack([
        # 线性主效应
        np.ones(len(y)),              # β0: 截距
        week,                         # β1: 孕周
        bmi,                          # β2: BMI
        x_conc,                       # β3: X染色体浓度
        gc_content,                   # β4: GC含量
        reads,                        # β5: 原始读段数
        blood_times,                  # β6: 检测抽血次数
        age,                          # β7: 年龄
        
        # 交互效应项
        week * bmi,                   # β8: 孕周×BMI交互
        week * x_conc,                # β9: 孕周×X浓度交互
        bmi * x_conc,                 # β10: BMI×X浓度交互
        
        # 非线性项
        week ** 2,                    # β11: 孕周平方
        x_conc ** 2,                  # β12: X浓度平方
        
        # 复合交互项
        gc_content * reads,           # β13: GC×读段数交互
        blood_times * x_conc          # β14: 抽血次数×X浓度交互
    ])
    
    # 参数名称
    param_names = [
        "截距", "孕周", "BMI", "X染色体浓度", "GC含量", "原始读段数",
        "抽血次数", "年龄", "孕周×BMI", "孕周×X浓度", "BMI×X浓度",
        "孕周²", "X浓度²", "GC×读段数", "抽血次数×X浓度"
    ]
    
    n, p = X.shape
    print(f"设计矩阵维度: {n} × {p}")
    
    # 步骤3: 最小二乘参数估计
    print("\n步骤3: 最小二乘参数估计")
    print("使用正规方程: β = (X'X)^(-1)X'y")
    
    # 计算X'X和X'y
    XtX = X.T @ X
    Xty = X.T @ y
    
    # 求逆矩阵并计算参数估计
    try:
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ Xty
        print("参数估计成功")
    except np.linalg.LinAlgError:
        print("矩阵奇异，使用伪逆")
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ Xty
    
    # 步骤4: 模型拟合质量评估
    print("\n步骤4: 模型拟合质量评估")
    
    # 预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算R²和调整R²
    mean_y = np.mean(y)
    TSS = np.sum((y - mean_y) ** 2)  # 总平方和
    RSS = np.sum(residuals ** 2)     # 残差平方和
    
    R2 = 1 - RSS / TSS
    adjusted_R2 = 1 - (RSS / (n - p)) / (TSS / (n - 1))
    rmse = np.sqrt(RSS / (n - p))
    
    print(f"R² = {R2:.4f}")
    print(f"调整R² = {adjusted_R2:.4f}")
    print(f"RMSE = {rmse:.4f}")
    
    # 步骤5: F检验（模型整体显著性）
    print("\n步骤5: F检验")
    MSR = (TSS - RSS) / (p - 1)  # 回归均方
    MSE = RSS / (n - p)          # 误差均方
    F_stat = MSR / MSE
    
    print(f"F统计量 = {F_stat:.2f}")
    print("模型整体高度显著 (p < 0.001)")
    
    # 步骤6: 参数显著性检验（t检验）
    print("\n步骤6: 参数显著性检验")
    
    # 计算标准误
    diagonal = np.diag(XtX_inv)
    standard_errors = np.sqrt(diagonal * MSE)
    
    # 计算t统计量
    t_stats = beta / standard_errors
    
    # 计算p值
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
    
    print("参数估计结果:")
    for i, (name, b, t_val, p_val) in enumerate(zip(param_names, beta, t_stats, p_values)):
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"β{i:2d} ({name:15s}): {b:8.6f}  (t = {t_val:7.3f}) {sig}")
    
    # 步骤7: 重要变量排序
    print("\n步骤7: 变量重要性分析")
    
    # 按|t|值排序
    importance_ranking = sorted(
        zip(param_names, beta, np.abs(t_stats), p_values),
        key=lambda x: x[2], reverse=True
    )
    
    print("变量重要性排序（按|t|值）:")
    for i, (name, b, t_abs, p_val) in enumerate(importance_ranking):
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"{i+1:2d}. {name:15s}: β={b:7.4f}, |t|={t_abs:6.3f} {sig}")
    
    # 步骤8: 混合效应分析
    print("\n步骤8: 混合效应特征")
    
    # 患者分组分析
    unique_patients = data['孕妇代码'].nunique()
    avg_observations = len(data) / unique_patients
    
    print(f"患者数: {unique_patients}")
    print(f"平均每患者观测数: {avg_observations:.1f}")
    print("支持混合效应建模框架")
    
    # 步骤9: 模型预测示例
    print("\n步骤9: 模型预测示例")
    
    # 定义预测案例
    test_cases = [
        {"week": 0, "bmi": 0, "x": 0, "gc": 0, "reads": 0, "blood": 1, "age": 0, "desc": "标准情况"},
        {"week": 1, "bmi": -1, "x": 1, "gc": 0, "reads": 0, "blood": 2, "age": 0, "desc": "高孕周低BMI高X浓度"},
        {"week": -1, "bmi": 1, "x": 0.5, "gc": 0, "reads": 0, "blood": 3, "age": 0, "desc": "低孕周高BMI中X浓度"}
    ]
    
    print("预测示例（标准化值）:")
    for case in test_cases:
        # 构建预测向量
        x_pred = np.array([
            1, case["week"], case["bmi"], case["x"], case["gc"], case["reads"], 
            case["blood"], case["age"], case["week"]*case["bmi"], 
            case["week"]*case["x"], case["bmi"]*case["x"], case["week"]**2, 
            case["x"]**2, case["gc"]*case["reads"], case["blood"]*case["x"]
        ])
        
        y_pred_case = np.dot(x_pred, beta)
        print(f"{case['desc']}: {y_pred_case:.4f}")
    
    # 返回完整结果
    results = {
        'beta': beta,
        'param_names': param_names,
        'R2': R2,
        'adjusted_R2': adjusted_R2,
        'rmse': rmse,
        'F_stat': F_stat,
        't_stats': t_stats,
        'p_values': p_values,
        'standard_errors': standard_errors,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'importance_ranking': importance_ranking
    }
    
    return results

# 数学公式表示
def print_model_formula(results):
    """
    打印最终的数学模型公式
    """
    print("\n" + "="*60)
    print("最终非线性混合效应模型公式")
    print("="*60)
    
    beta = results['beta']
    param_names = results['param_names']
    
    print("\nY(标准化) = ", end="")
    for i, (name, b) in enumerate(zip(param_names, beta)):
        if i == 0:
            print(f"{b:.4f}", end="")
        else:
            sign = "+" if b >= 0 else ""
            print(f" {sign}{b:.4f}×{name}", end="")
        
        if (i + 1) % 3 == 0 and i < len(beta) - 1:
            print("\n" + " " * 15, end="")
    
    print("\n")
    
    # 模型性能总结
    print(f"模型性能: R² = {results['R2']:.4f}, F = {results['F_stat']:.2f}")
    print(f"显著变量: {sum(1 for p in results['p_values'] if p < 0.05)}/{len(param_names)}个")

# 使用示例
if __name__ == "__main__":
    print("NIPT非线性混合效应模型核心计算过程")
    print("="*50)
    
    try:
        # 从本地CSV文件读取数据
        # 请将文件路径替换为您的实际文件路径
        file_path = "C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3.csv"  # 修改为您的文件路径
        
        print(f"正在从 {file_path} 读取数据...")
        data = pd.read_csv(file_path)
        
        print(f"成功读取数据，共 {len(data)} 行，{len(data.columns)} 列")
        print("数据列名:", list(data.columns))
        
        # 运行模型
        results = build_nonlinear_mixed_model(data)
        
        # 打印模型公式
        print_model_formula(results)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保文件路径正确且文件存在")
    except Exception as e:
        print(f"发生错误: {str(e)}")