import pandas as pd, numpy as np, pulp as pl, math

# ========== 0. 全局常量 (保持不变) ===============================================
RB_TOTAL = 50
PERIOD_MS = 100
N_PERIOD = 10
B = 360e3
# --- 调整权重以反映优先级: URLLC >> eMBB > mMTC ---
W_URLLC = 101.0
W_eMBB = 11.0
W_mMTC = 1.0
R_SLA_eMBB = 10.0  # eMBB 单用户速率目标 (Mbps)

# ========== 1. 读取和预处理 (保持不变) =============================
path = r'B题\附件\附件2\channel_data.xlsx'
try:
    L_df = pd.read_excel(path, sheet_name='大规模衰减').drop(columns='Time')
    h_df = pd.read_excel(path, sheet_name='小规模瑞丽衰减').drop(columns='Time')
    traf_df = pd.read_excel(path, sheet_name='用户任务流').drop(columns='Time')
except FileNotFoundError:
    print(f"文件未找到: {path}")
    exit()

traf_df *= 1e6  # Mbit -> bit

users = L_df.columns.tolist()
u_type = {u: ('URLLC' if u.startswith('U') else 'eMBB' if u.startswith('e') else 'mMTC') for u in users}

P_tx_dBm, N0_dBmHz = 23, -174
Pt_mW = 10 ** (P_tx_dBm / 10)
N0_RB_dBm = N0_dBmHz + 10 * math.log10(B)
noise_mW = 10 ** (N0_RB_dBm / 10)

snr_df = pd.DataFrame(index=L_df.index, columns=users)
for u in users:
    Pr_mW = Pt_mW * (h_df[u].abs() ** 2) / (10 ** (L_df[u] / 10))
    snr_df[u] = Pr_mW / noise_mW

# ========== 2. 初始化 (增加一个用于调试的记录) =====================================
q_bits = {u: 0 for u in users}
delay = {u: 0 for u in users}
utility_total = 0
alloc_record = []
utility_record = []  # 记录每轮的效用

# ========== 3. 主循环 (10 周期) ========================================
print("开始进行10轮动态资源分配...")
for p in range(N_PERIOD):
    t = p * PERIOD_MS
    snr_row = snr_df.iloc[t]
    arrivals = traf_df.iloc[t:t + PERIOD_MS].sum()
    for u in users:
        q_bits[u] += arrivals[u]

    # ---------- U_tab 效用表计算 (核心修正区域) ----------
    U = {s: [0.0] * (RB_TOTAL + 1) for s in ['URLLC', 'eMBB', 'mMTC']}

    # 筛选出当前有任务的用户
    urllc_wait_users = [u for u in users if u_type[u] == 'URLLC' and q_bits[u] > 0]
    embb_wait_users = [u for u in users if u_type[u] == 'eMBB' and q_bits[u] > 0]
    mmtc_wait_users = [u for u in users if u_type[u] == 'mMTC' and q_bits[u] > 0]

    for k in range(1, RB_TOTAL + 1):  # k from 1 to 50
        # --- URLLC 效用计算 (时延敏感) ---
        if urllc_wait_users:
            # 假设 k 个RB公平地分给所有等待的URLLC用户
            rb_per_user_L = k / len(urllc_wait_users)
            total_sent_bits_L = 0
            for u in urllc_wait_users:
                # 计算每个用户能获得的速率
                rate_u = rb_per_user_L * B * math.log2(1 + snr_row[u])
                # 计算100ms内能传输的数据量
                sent_bits = rate_u * (PERIOD_MS / 1000)
                total_sent_bits_L += min(q_bits[u], sent_bits)
            # 效用 = 权重 * 总吞吐量 (Mbits)。时延惩罚在后面计算
            U['URLLC'][k] = W_URLLC * (total_sent_bits_L / 1e6)

        # --- eMBB 效用计算 (速率敏感) ---
        if embb_wait_users:
            # 假设 k 个RB公平地分给所有等待的eMBB用户
            rb_per_user_B = k / len(embb_wait_users)
            total_satisfaction_B = 0
            for u in embb_wait_users:
                # 计算每个用户能获得的速率 (Mbps)
                rate_u_mbps = (rb_per_user_B * B * math.log2(1 + snr_row[u])) / 1e6
                # 效用 = 用户的速率满意度 (0-1之间)
                satisfaction = np.log1p(rate_u_mbps) / np.log1p(R_SLA_eMBB)  # 归一化对数效用
                total_satisfaction_B += min(1.0, satisfaction)
            # 效用 = 权重 * 总满意度
            U['eMBB'][k] = W_eMBB * total_satisfaction_B

        # --- mMTC 效用计算 (连接数敏感) ---
        if mmtc_wait_users:
            # 假设每个mMTC用户固定需要1个RB来完成信令交互
            rb_per_user_M = 1
            # 计算用k个RB最多能服务的用户数
            served_count = k // rb_per_user_M
            # 效用 = 权重 * 成功服务的用户数 (与等待数取小)
            U['mMTC'][k] = W_mMTC * min(len(mmtc_wait_users), served_count)

    # ---------- MILP (逻辑不变，但现在输入是合理的U_tab) ----------
    prob = pl.LpProblem(f"P{p}", pl.LpMaximize)
    # 使用字典推导式创建变量，更简洁
    x = {(s, k): pl.LpVariable(f"x_{s}_{k}", cat='Binary') for s in U for k in range(RB_TOTAL + 1)}

    # 目标函数
    prob += pl.lpSum(U[s][k] * x[s, k] for s, k_list in U.items() for k, u_val in enumerate(k_list))

    # 约束1: 只能选择一个 (s, k) 组合
    prob += pl.lpSum(x[s, k] for s in U for k in range(RB_TOTAL + 1)) == 1

    # 约束2: 总RB数必须是50 (通过模型结构隐式保证，但最好显式写出)
    prob += pl.lpSum(k * x[s, k] for s in U for k in range(RB_TOTAL + 1)) == RB_TOTAL

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    # 解析结果
    k_s = {s: 0 for s in U}
    total_rb_allocated = 0
    for s in U:
        for k in range(RB_TOTAL + 1):
            if pl.value(x[s, k]) == 1:
                k_s[s] = k
                total_rb_allocated += k

    # 为了应对可能的空解，如果总和不是50，说明有问题，需要回退方案
    if total_rb_allocated != RB_TOTAL:
        # Fallback: 如果MILP无解，则按权重比例分配
        total_utility_estimate = [sum(U[s]) for s in U]
        total_sum = sum(total_utility_estimate)
        if total_sum > 0:
            k_s['URLLC'] = round(RB_TOTAL * total_utility_estimate[0] / total_sum)
            k_s['eMBB'] = round(RB_TOTAL * total_utility_estimate[1] / total_sum)
            k_s['mMTC'] = RB_TOTAL - k_s['URLLC'] - k_s['eMBB']
        else:  # 如果所有效用都是0
            k_s = {'URLLC': 17, 'eMBB': 17, 'mMTC': 16}

    alloc_record.append(k_s)
    current_utility = sum(U[s][int(k_s[s])] for s in k_s)
    utility_record.append(current_utility)

    # ---------- 队列推进 & 效用累计 (更精细的计算) ----------
    # URLLC
    if urllc_wait_users and k_s['URLLC'] > 0:
        rb_per_user_L = k_s['URLLC'] / len(urllc_wait_users)
        for u in urllc_wait_users:
            rate_u = rb_per_user_L * B * math.log2(1 + snr_row[u])
            served = rate_u * (PERIOD_MS / 1000)
            q_bits[u] = max(0, q_bits[u] - served)
    # eMBB
    if embb_wait_users and k_s['eMBB'] > 0:
        rb_per_user_B = k_s['eMBB'] / len(embb_wait_users)
        for u in embb_wait_users:
            rate_u = rb_per_user_B * B * math.log2(1 + snr_row[u])
            served = rate_u * (PERIOD_MS / 1000)
            q_bits[u] = max(0, q_bits[u] - served)
    # mMTC
    if mmtc_wait_users and k_s['mMTC'] > 0:
        rb_per_user_M = 1
        served_count = int(k_s['mMTC'] // rb_per_user_M)
        for i, u in enumerate(mmtc_wait_users):
            if i < served_count:
                q_bits[u] = 0  # mMTC任务小，假设分配到RB就能完成

    # 更新时延和计算惩罚
    final_utility_this_round = current_utility
    for u in users:
        if q_bits[u] > 0:
            delay[u] += PERIOD_MS
            if u_type[u] == 'URLLC' and delay[u] > 20:  # URLLC时延阈值20ms
                final_utility_this_round -= 5  # 失败惩罚
                delay[u] = 0  # 重置以避免重复惩罚
        else:
            delay[u] = 0
    utility_total += final_utility_this_round

# ========== 4. 输出 =====================================================
print("\n10 轮切片 RB 分配（URLLC, eMBB, mMTC）:")
for i, ks in enumerate(alloc_record):
    print(f"  周期 {i}: U={ks['URLLC']}, E={ks['eMBB']}, M={ks['mMTC']} (效用: {utility_record[i]:.2f})")

print(f"\n累计用户服务质量 Utility = {utility_total:.2f}")