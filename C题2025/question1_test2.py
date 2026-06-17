import numpy as np      #非线性混合效应（NLME）
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 读数据 -------------------------------------------------------------
df = pd.read_csv('附件_wash.csv').dropna()
y = df['Y染色体浓度'].values
x = df['X染色体浓度'].values
# 构造个体列（若已有孕妇ID可替换）
df['id'] = df.groupby(['检测抽血次数','孕妇BMI','原始读段数','GC含量','检测孕周']).ngroup()
pid = df['id'].astype(int).values
n_id = df['id'].nunique()

# 2. 曲线定义 -----------------------------------------------------------
def f(x, b1, b2):
    return b1 * x / (b2 + x)          # Michaelis-Menten

# 3. 个体级负对数似然（Laplace）----------------------------------------
def nll_i(i, theta, x, y, pid):
    b1, b2, log_psi, log_sigma = theta
    # 硬边界：psi ∈ [1e-6, 1e6]，sigma ∈ [1e-6, 1e2]
    if log_psi > 20 or log_psi < -20 or log_sigma > 10 or log_sigma < -20:
        return 1e10
    psi   = np.exp(log_psi)   + 1e-8
    sigma = np.exp(log_sigma) + 1e-8
    idx   = np.where(pid == i)[0]
    xi, yi = x[idx], y[idx]

    def nll_b(b):
        mu = f(xi, b1 + b, b2)
        return 0.5*np.sum((yi - mu)**2)/sigma**2 + 0.5*b**2/psi

    b_hat = minimize(nll_b, 0., method='L-BFGS-B', bounds=[(-1e3, 1e3)]).x[0]
    hess_b = 1./psi + np.sum((f(xi, b1 + b_hat, b2) - yi)**2) / sigma**2
    return nll_b(b_hat) + 0.5*np.log(2*np.pi/hess_b)

# 4. 总负对数似然 --------------------------------------------------------
def nll_total(theta):
    return sum(nll_i(i, theta, x, y, pid) for i in range(n_id))

# 5. 有界优化 ------------------------------------------------------------
init = [y.max(), x.mean(), -2., -2.]          # b1,b2,log_psi,log_sigma
bnds = [(0, None), (0, None), (-20, 20), (-20, 10)]
res  = minimize(nll_total, init, method='L-BFGS-B', bounds=bnds)
b1, b2, psi, sigma = res.x[0], res.x[1], np.exp(res.x[2]), np.exp(res.x[3])

# 6. 经验贝叶斯预测 b_i --------------------------------------------------
b_hat = np.zeros(n_id)
for i in range(n_id):
    idx = np.where(pid == i)[0]
    xi = x[idx]
    psi_i = psi + 1e-8
    sigma_i = sigma + 1e-8
    def nll_b(b): return 0.5*np.sum((y[idx]-f(xi,b1+b,b2))**2)/sigma_i**2 + 0.5*b**2/psi_i
    b_hat[i] = minimize(nll_b, 0., method='L-BFGS-B', bounds=[(-1e3, 1e3)]).x[0]

# 7. 评价 ---------------------------------------------------------------
y_pred_marg = f(x, b1, b2)
y_pred_cond = f(x, b1 + b_hat[pid], b2)
RMSE_marg = np.sqrt(np.mean((y - y_pred_marg)**2))
RMSE_cond = np.sqrt(np.mean((y - y_pred_cond)**2))
R2_marg = 1 - np.sum((y - y_pred_marg)**2) / np.sum((y - y.mean())**2)
R2_cond = 1 - np.sum((y - y_pred_cond)**2) / np.sum((y - y.mean())**2)

print('固定效应 MLE：')
print('b1 = %.4f   b2 = %.4f   psi = %.4f   sigma = %.4f' % (b1, b2, psi, sigma))
print('边际 R² = %.4f    条件 R² = %.4f' % (R2_marg, R2_cond))
print('边际 RMSE = %.4f  条件 RMSE = %.4f' % (RMSE_marg, RMSE_cond))

# 8. 残差图 -------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_marg, y - y_pred_marg, alpha=0.6)
plt.axhline(0, color='red', ls='--'); plt.title('边际残差')
plt.xlabel('拟合值'); plt.ylabel('残差')

plt.subplot(1, 2, 2)
plt.scatter(y_pred_cond, y - y_pred_cond, alpha=0.6)
plt.axhline(0, color='red', ls='--'); plt.title('条件残差')
plt.xlabel('拟合值'); plt.ylabel('残差')
plt.tight_layout(); plt.show()