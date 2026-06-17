import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 读数据 ------------------------------------------------------------
df = pd.read_csv('附件_wash.csv').dropna()
y = df['Y染色体浓度'].values
x = df['X染色体浓度'].values

# 2. 定义曲线（可任意改）-----------------------------------------------
def f(x, b1, b2):
    """Michaelis-Menten 固定效应曲线"""
    return b1 * x / (b2 + x)

# 3. 负对数似然（独立同分布正态误差）-----------------------------------
def nll(theta):
    b1, b2, log_sigma = theta
    sigma = np.exp(log_sigma)
    yhat = f(x, b1, b2)
    return 0.5 * len(y) * np.log(2 * np.pi * sigma**2) + 0.5 * np.sum((y - yhat)**2) / sigma**2

# 4. 初始值 & 优化 -----------------------------------------------------
init = [y.max(), x.mean(), np.log(0.1)]          # b1,b2,log_sigma
res  = minimize(nll, init, method='BFGS')
b1, b2, sigma = res.x[0], res.x[1], np.exp(res.x[2])

# 5. 标准误（数值 Hessian 逆对角线）------------------------------------
from scipy.optimize import approx_fprime
eps = 1e-8
hess = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ei = np.zeros(3); ej = np.zeros(3)
        ei[i] = eps; ej[j] = eps
        hess[i, j] = (nll(res.x + ei + ej) - nll(res.x + ei) - nll(res.x + ej) + nll(res.x)) / eps**2
se = np.sqrt(np.diag(np.linalg.inv(hess)))
print('固定效应 MLE：')
print('b1 = %.4f ± %.4f' % (b1, se[0]))
print('b2 = %.4f ± %.4f' % (b2, se[1]))
print('sigma = %.4f ± %.4f' % (sigma, se[2]))

# 6. RMSE & R² ---------------------------------------------------------
yhat = f(x, b1, b2)
rmse = np.sqrt(np.mean((y - yhat)**2))
r2   = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)
print('\n训练集 RMSE = %.4f' % rmse)
print('R² = %.4f' % r2)

# 7. 画图 --------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(x, y, alpha=0.6, label='观测')
x_smooth = np.linspace(x.min(), x.max(), 200)
plt.plot(x_smooth, f(x_smooth, b1, b2), 'r-', label='MM 曲线')
plt.xlabel('X染色体浓度')
plt.ylabel('Y染色体浓度')
plt.title('固定效应非线性拟合')
plt.legend()
plt.show()