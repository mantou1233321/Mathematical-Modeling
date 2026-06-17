import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error

# 1. 读数据
data = pd.read_csv('C:/Users/26218/Desktop/2025/2025题目/C题/附件_wash_3.csv')
data = data.dropna()

# 2. 定义 X, y
y = data['Y染色体浓度']
X = data[['检测抽血次数', '孕妇BMI', '原始读段数', 'GC含量', 'X染色体浓度','检测孕周']]

# 3. 通用评估函数
def rmse_score(model, X, y):
    return np.sqrt(-cross_val_score(model, X, y,
                                   cv=KFold(30, shuffle=True, random_state=42),
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1).mean())

# 4. 模型池
models = {
    'OLS (Robust)'     : make_pipeline(StandardScaler(),
                                      sm.OLS(y, sm.add_constant(X)).fit()),
    'Ridge'            : make_pipeline(StandardScaler(),
                                      RidgeCV(alphas=np.logspace(-3, 3, 100), cv=30)),
    'Lasso'            : make_pipeline(StandardScaler(),
                                      LassoCV(alphas=np.logspace(-3, 3, 100), cv=30, max_iter=2000)),
    'ElasticNet'       : make_pipeline(StandardScaler(),
                                      ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                                  alphas=np.logspace(-3, 3, 100), cv=30, max_iter=2000)),
    'RandomForest'     : RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
    'GradientBoosting' : GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                                   max_depth=3, random_state=42),
}

# 5. 跑模型
results = {}
for name, model in models.items():
    if name == 'OLS (Robust)':
        ols = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3')
        rmse = np.sqrt(ols.mse_resid)          # 训练集 RMSE，仅参考
        r2   = ols.rsquared_adj
        coef = ols.params
    else:
        rmse = rmse_score(model, X, y)
        r2   = cross_val_score(model, X, y,
                              cv=KFold(10, shuffle=True, random_state=42),
                              scoring='r2').mean()
        coef = None
    results[name] = {'RMSE': rmse, 'R2': r2, 'coef': coef}
    print(f"{name:15s} | RMSE={rmse:.4f} | R2={r2:.4f}")

# 6. 变量重要度（树模型）
gb = models['GradientBoosting'].fit(X, y)
importances = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nGradientBoosting 变量重要度：")
print(importances)