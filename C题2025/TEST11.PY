"""
NIPT非线性混合效应模型核心计算过程
基于标准化数据的Y染色体浓度预测
包含固定效应和随机效应的混合模型
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

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
    required_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI', '孕妇代码']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少必要的列: {missing_columns}")
    
    y = data['Y染色体浓度'].values  # 因变量（标准化）
    
    # 提取标准化的自变量
    week = data['检测孕周'].values
    bmi = data['孕妇BMI'].values
    patient_ids = data['孕妇代码'].values
    
    print(f"样本数: {len(y)}")
    print(f"患者数: {len(np.unique(patient_ids))}")
    print(f"变量数: 2个主要变量（检测孕周、孕妇BMI）的5个多项式项")
    
    # 步骤2: 构建设计矩阵
    print("\n步骤2: 构建设计矩阵")
    
    # 固定效应设计矩阵
    X_fixed = np.column_stack([
        np.ones(len(y)),              # β0: 截距
        week,                         # β1: 孕周
        bmi,                          # β2: BMI
        week ** 2,                    # β3: 孕周平方
        week ** 3,                    # β4: 孕周立方
        bmi ** 2,                     # β5: BMI平方
    ])
    
    # 随机效应设计矩阵 (假设截距有随机效应)
    Z = np.ones(len(y))  # 随机截距
    
    # 参数名称
    fixed_param_names = ["截距", "孕周", "BMI", "孕周²", "孕周³", "BMI²"]
    random_param_names = ["随机截距"]
    
    n, p_fixed = X_fixed.shape
    print(f"固定效应设计矩阵维度: {n} × {p_fixed}")
    
    # 步骤3: 定义混合效应模型的对数似然函数
    def mixed_model_loglikelihood(theta, X, Z, y, patient_ids):
        """
        混合效应模型的对数似然函数
        theta: [固定效应参数, 随机效应方差, 残差方差]
        """
        # 提取参数
        beta = theta[:p_fixed]  # 固定效应参数
        sigma_b = theta[p_fixed]  # 随机效应标准差
        sigma_e = theta[p_fixed + 1]  # 残差标准差
        
        # 计算固定效应部分
        fixed_effects = X @ beta
        
        # 计算每个患者的随机效应贡献
        loglik = 0
        unique_patients = np.unique(patient_ids)
        
        for patient in unique_patients:
            patient_mask = (patient_ids == patient)
            y_patient = y[patient_mask]
            fixed_patient = fixed_effects[patient_mask]
            n_patient = np.sum(patient_mask)
            
            # 对于每个患者，积分掉随机效应（使用拉普拉斯近似）
            # 这里简化处理，使用条件似然
            patient_variance = sigma_b**2 + sigma_e**2
            residuals = y_patient - fixed_patient
            patient_loglik = -0.5 * (n_patient * np.log(2 * np.pi * patient_variance) + 
                                    np.sum(residuals**2) / patient_variance)
            
            loglik += patient_loglik
        
        return -loglik  # 返回负对数似然用于最小化
    
    # 步骤4: 参数估计
    print("\n步骤4: 混合效应模型参数估计")
    
    # 初始参数估计（使用固定效应模型）
    XtX = X_fixed.T @ X_fixed
    Xty = X_fixed.T @ y
    try:
        beta_init = np.linalg.solve(XtX, Xty)
    except:
        beta_init = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    
    # 初始参数：固定效应 + 随机效应标准差 + 残差标准差
    theta_init = np.concatenate([beta_init, [0.1, 0.5]])  # 初始猜测
    
    # 使用数值优化估计参数
    bounds = [(None, None)] * p_fixed + [(1e-6, None), (1e-6, None)]
    
    result = optimize.minimize(
        mixed_model_loglikelihood,
        theta_init,
        args=(X_fixed, Z, y, patient_ids),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if result.success:
        print("混合效应模型参数估计成功")
        theta_opt = result.x
        beta_opt = theta_opt[:p_fixed]
        sigma_b = theta_opt[p_fixed]
        sigma_e = theta_opt[p_fixed + 1]
    else:
        print("优化失败，使用固定效应估计")
        beta_opt = beta_init
        sigma_b = 0.1
        sigma_e = 0.5
    
    print(f"随机效应标准差: {sigma_b:.4f}")
    print(f"残差标准差: {sigma_e:.4f}")
    
    # 步骤5: 模型拟合质量评估
    print("\n步骤5: 模型拟合质量评估")
    
    # 预测值（仅固定效应部分）
    y_pred_fixed = X_fixed @ beta_opt
    residuals = y - y_pred_fixed
    
    # 计算条件R²（仅固定效应部分）
    mean_y = np.mean(y)
    TSS = np.sum((y - mean_y) ** 2)
    RSS = np.sum(residuals ** 2)
    
    R2_conditional = 1 - RSS / TSS
    rmse = np.sqrt(RSS / (n - p_fixed))
    
    print(f"条件R² (仅固定效应): {R2_conditional:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"随机效应方差比例: {sigma_b**2/(sigma_b**2 + sigma_e**2):.4f}")
    
    # 步骤6: 参数显著性检验
    print("\n步骤6: 参数显著性检验")
    
    # 使用固定效应模型的协方差矩阵近似
    try:
        # 计算Hessian矩阵近似
        hessian = np.linalg.inv(XtX) * (n / (n - p_fixed)) * (RSS / n)
        standard_errors = np.sqrt(np.diag(hessian))
        t_stats = beta_opt / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p_fixed))
        
        print("固定效应参数估计结果:")
        for i, (name, b, t_val, p_val) in enumerate(zip(fixed_param_names, beta_opt, t_stats, p_values)):
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"β{i:2d} ({name:15s}): {b:8.6f}  (t = {t_val:7.3f}) {sig}")
    except:
        print("无法计算标准误，显示参数估计值:")
        for i, (name, b) in enumerate(zip(fixed_param_names, beta_opt)):
            print(f"β{i:2d} ({name:15s}): {b:8.6f}")
    
    # 步骤7: 随机效应分析
    print("\n步骤7: 随机效应分析")
    
    # 估计每个患者的随机效应（BLUP）
    unique_patients = np.unique(patient_ids)
    random_effects = {}
    
    for patient in unique_patients:
        patient_mask = (patient_ids == patient)
        y_patient = y[patient_mask]
        fixed_patient = y_pred_fixed[patient_mask]
        
        # 简单估计随机效应
        re_estimate = np.mean(y_patient - fixed_patient)
        random_effects[patient] = re_estimate
    
    print(f"随机效应范围: [{min(random_effects.values()):.4f}, {max(random_effects.values()):.4f}]")
    print(f"随机效应标准差: {np.std(list(random_effects.values())):.4f}")
    
    # 步骤8: 模型验证
    print("\n步骤8: 模型验证")
    
    # 计算组内相关系数 (ICC)
    icc = sigma_b**2 / (sigma_b**2 + sigma_e**2)
    print(f"组内相关系数 (ICC): {icc:.4f}")
    
    # 步骤9: 模型预测
    print("\n步骤9: 模型预测示例")
    
    test_cases = [
        {"week": 0, "bmi": 0, "desc": "标准情况"},
        {"week": 1, "bmi": -1, "desc": "高孕周低BMI"},
        {"week": -1, "bmi": 1, "desc": "低孕周高BMI"}
    ]
    
    print("固定效应预测（标准化值）:")
    for case in test_cases:
        x_pred = np.array([1, case["week"], case["bmi"], 
                          case["week"]**2, case["week"]**3, case["bmi"]**2])
        y_pred = np.dot(x_pred, beta_opt)
        print(f"{case['desc']}: {y_pred:.4f}")
    
    # 返回完整结果
    results = {
        'beta_fixed': beta_opt,
        'sigma_b': sigma_b,
        'sigma_e': sigma_e,
        'fixed_param_names': fixed_param_names,
        'R2_conditional': R2_conditional,
        'rmse': rmse,
        'icc': icc,
        'random_effects': random_effects,
        'y_pred_fixed': y_pred_fixed,
        'residuals': residuals,
        'patient_ids': patient_ids
    }
    
    return results

def print_mixed_model_formula(results):
    """
    打印混合效应模型公式
    """
    print("\n" + "="*60)
    print("非线性混合效应模型公式")
    print("="*60)
    
    beta = results['beta_fixed']
    param_names = results['fixed_param_names']
    
    print("\nY_ij = ", end="")
    for i, (name, b) in enumerate(zip(param_names, beta)):
        if i == 0:
            print(f"{b:.4f}", end="")
        else:
            sign = "+" if b >= 0 else ""
            print(f" {sign}{b:.4f}×{name}_ij", end="")
    
    print(f" + b_i + ε_ij")
    print("\n其中:")
    print(f"b_i ~ N(0, {results['sigma_b']**2:.6f})  # 随机截距")
    print(f"ε_ij ~ N(0, {results['sigma_e']**2:.6f})  # 残差")
    print(f"b_i 和 ε_ij 相互独立")
    
    print(f"\n模型性能:")
    print(f"条件R² = {results['R2_conditional']:.4f}")
    print(f"ICC = {results['icc']:.4f}")
    print(f"RMSE = {results['rmse']:.4f}")

# 使用示例
if __name__ == "__main__":
    print("NIPT非线性混合效应模型核心计算过程")
    print("包含固定效应和随机效应的混合模型")
    print("="*60)
    
    try:
        # 从本地CSV文件读取数据
        file_path = "C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_only3.csv"
        
        print(f"正在从 {file_path} 读取数据...")
        data = pd.read_csv(file_path)
        
        print(f"成功读取数据，共 {len(data)} 行，{len(data.columns)} 列")
        
        # 运行混合效应模型
        results = build_nonlinear_mixed_model(data)
        
        # 打印模型公式
        print_mixed_model_formula(results)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()