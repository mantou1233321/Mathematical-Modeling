import pandas as pd, numpy as np, pulp as pl, math

# === 1. 读取数据 & 整形成“长表” =================================================
raw = pd.read_excel(r'B题\附件\附件1\channel_data.xlsx')
raw.columns = raw.columns.str.strip()          # 去空格
user_cols = [c for c in raw.columns if c != 'Time']

records = []
for col in user_cols:
    s  = raw.loc[0, col]                      # 题目仅给一个时隙
    typ= 'URLLC' if col.startswith('U') else 'eMBB' if col.startswith('e') else 'mMTC'
    records.append({'user': col, 'type': typ, 'snr': s})
df = pd.DataFrame(records)

# === 2. 预先计算“每个用户分 k 个 RB 的效用” =====================================
B       = 360e3                 # Hz
R_req   = 10                    # Mbps
W       = {'URLLC':3, 'eMBB':1, 'mMTC':0.5}

U = {}                          # (u,k) -> utility
for idx, row in df.iterrows():
    for k in range(51):         # 0..50 RB
        rate = k*B*math.log2(1+row.snr)/1e6   # Mbps
        if row.type == 'URLLC':
            util = W['URLLC']*min(rate, R_req)
        elif row.type == 'eMBB':
            util = W['eMBB']*math.log1p(rate)
        else:
            util = W['mMTC']*math.sqrt(rate)
        U[(idx,k)] = util

# === 3. 构建 MILP (0-1 选择模型) ===============================================
prob = pl.LpProblem("RB_Allocation", pl.LpMaximize)

x = {(i,k): pl.LpVariable(f"x_{i}_{k}", cat='Binary')
     for i in df.index for k in range(51)}

# 目标函数
prob += pl.lpSum(U[(i,k)]*x[(i,k)] for i in df.index for k in range(51))

# 每个用户只能选一个 k
for i in df.index:
    prob += pl.lpSum(x[(i,k)] for k in range(51)) == 1

# RB 总量 = 50
prob += pl.lpSum(k*x[(i,k)] for i in df.index for k in range(51)) == 50

prob.solve(pl.PULP_CBC_CMD(msg=False))

# === 4. 整理输出 ===============================================================
alloc_k = []
for i in df.index:
    k = next(k for k in range(51) if pl.value(x[(i,k)])==1)
    alloc_k.append(k)

df['n_RB'] = alloc_k
df['Rate(Mbps)'] = [k*B*math.log2(1+df.loc[i,'snr'])/1e6 for i,k in zip(df.index,alloc_k)]
df['Utility'] = [U[(i,k)] for i,k in zip(df.index,alloc_k)]

print(df[['user','type','n_RB','Rate(Mbps)','Utility']])
print("\nTotal Utility =", df['Utility'].sum())
