import numpy as np
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

# 2. Michaelis-Menten 曲线 ---------------------------------------------
def f(x, b1, b2):
    return b1 * x / (b2 + x)

# 3. Huber 损失及其梯度 --------------------------------------------------
def huber(r, delta=1.345):
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.zeros_like(r)
    loss[mask] = 0.5 * r[mask]**2
    loss[~mask] = delta * abs_r[~mask] - 0.5 * delta**2
    return loss.sum()

# 4. 总目标函数（Huber 残差）---------------------------------------------
def obj(theta, delta=1.345):
    b1, b2 = theta
    r = y - f(x, b1, b2)
    return huber(r, delta)

# 5. 估计 ---------------------------------------------------------------
# 5.1 先跑一遍 OLS 拿 σ，再设 δ = 1.345σ
ols_res = minimize(lambda b: np.sum((y - f(x, b[0], b[1]))**2), [y.max(), x.mean()], method='BFGS')
delta0 = 1.345 * np.sqrt(np.mean((y - f(x, *ols_res.x))**2))

# 5.2 Huber 稳健拟合
res = minimize(obj, ols_res.x, args=(delta0,), method='L-BFGS-B', bounds=[(0, None), (0, None)])
b1_huber, b2_huber = res.x

# 6. 评价 ---------------------------------------------------------------
y_pred = f(x, b1_huber, b2_huber)
rmse = np.sqrt(np.mean((y - y_pred)**2))
r2   = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

print('Huber 稳健 MM 拟合：')
print('b1 = %.4f   b2 = %.4f   δ = %.4f' % (b1_huber, b2_huber, delta0))
print('RMSE = %.4f   R² = %.4f' % (rmse, r2))

# 7. 残差图 -------------------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3))
plt.scatter(y_pred, y - y_pred, alpha=0.6)
plt.axhline(0, color='red', ls='--')
plt.xlabel('拟合值'); plt.ylabel('残差'); plt.title('Huber 残差图')
plt.show()