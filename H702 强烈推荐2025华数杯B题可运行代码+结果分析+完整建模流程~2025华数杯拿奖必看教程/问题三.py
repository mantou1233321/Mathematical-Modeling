#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
import itertools, os

# ----------------- 常量 -----------------
RB_TOTAL = 50
RB_PER_SLICE = {'URLLC': 10, 'eMBB': 5, 'mMTC': 2}
SLA_RATE = {'URLLC': 10e6, 'eMBB': 50e6, 'mMTC': 1e6}     # bit s-1
ALPHA = 0.95                                               # URLLC 满分权重

BANDWIDTH_RB = 360e3                                       # 360 kHz
THERMAL_DBM  = -174                                        # dBm Hz-1
NOISE_FIG_DB = 7
DT           = 0.1                                         # 100 ms

# ----------------- 数据读取 -----------------
DATA_DIR = r'B题\附件\附件3'        # ← 如路径有变请修改
df_bs = {i: pd.read_excel(os.path.join(DATA_DIR, f'BS{i}.xlsx'))
         for i in (1, 2, 3)}

URLLC = [f'U{i}' for i in range(1, 7)]
eMBB  = [f'e{i}' for i in range(1, 13)]
mMTC  = [f'm{i}' for i in range(1, 31)]
USER_CLASS = {**{u: 'URLLC' for u in URLLC},
              **{u: 'eMBB'  for u in eMBB},
              **{u: 'mMTC'  for u in mMTC}}

# ----------------- 工具函数 -----------------
def dbm2mw(p_dbm):  return 10 ** ((p_dbm - 30) / 10)
def sinr(loss_db, p_dbm):
    """单基站 SINR（忽略小区间干扰示例版）"""
    n0_mw  = dbm2mw(THERMAL_DBM + NOISE_FIG_DB) * BANDWIDTH_RB
    sig_mw = dbm2mw(p_dbm - loss_db)
    return sig_mw / n0_mw

def rate(rb, sinr_val):           # bit s-1
    return rb * BANDWIDTH_RB * np.log2(1 + sinr_val)

# --------- 切片-RB 贪心分配 ----------
def greedy_rb(users, gain_db, p_dbm, rb_total=RB_TOTAL):
    alloc, rest = {u: 0 for u in users}, rb_total

    # 1) URLLC
    for u in users:
        if USER_CLASS[u] == 'URLLC' and rest >= RB_PER_SLICE['URLLC']:
            alloc[u] = RB_PER_SLICE['URLLC'];  rest -= RB_PER_SLICE['URLLC']

    # 2) eMBB（按信道好坏排序）
    embb = [u for u in users if USER_CLASS[u] == 'eMBB']
    for u in sorted(embb, key=lambda x: -gain_db[x]):
        if rest >= RB_PER_SLICE['eMBB']:
            alloc[u] = RB_PER_SLICE['eMBB'];  rest -= RB_PER_SLICE['eMBB']

    # 3) mMTC 轮询
    for u in [u for u in users if USER_CLASS[u] == 'mMTC']:
        if rest >= RB_PER_SLICE['mMTC']:
            alloc[u] = RB_PER_SLICE['mMTC'];  rest -= RB_PER_SLICE['mMTC']
        else:
            break
    return alloc

def qos(alloc, gain_db, p_dbm):
    total = 0.0
    for u, rb in alloc.items():
        if rb == 0: continue
        gamma = sinr(gain_db[u], p_dbm)
        cls   = USER_CLASS[u]

        if cls == 'eMBB':
            R = rate(rb, gamma)
            total += 1 if R >= SLA_RATE['eMBB'] else R / SLA_RATE['eMBB']
        elif cls == 'URLLC':
            R = rate(rb, gamma)
            total += ALPHA if R >= SLA_RATE['URLLC'] else 0          # 未达标不给分
        else:  # mMTC
            R = rate(rb, gamma)
            total += 1 if R >= SLA_RATE['mMTC'] else 0
    return total

def best_power(users, gain_db):
    best_q, best_p, best_alloc = -1, 10, {}
    for p in range(10, 31, 2):                  # 10–30 dBm
        alloc = greedy_rb(users, gain_db, p)
        q     = qos(alloc, gain_db, p)
        if q > best_q:
            best_q, best_p, best_alloc = q, p, alloc
    return best_p, best_alloc, best_q

# ----------------- 主循环 -----------------
def main():
    results = []
    for k in range(10):                         # 0–9 时隙
        idx = int(k * DT * 1000)                # 对应 1 ms 行号
        # ---- (A) 关联：每用户连路径损耗最小的 BS ----
        assoc = {1: [], 2: [], 3: []}
        for u in USER_CLASS:
            losses = {b: df_bs[b].iloc[idx][u] for b in (1, 2, 3)}
            best_b = min(losses, key=losses.get)
            assoc[best_b].append(u)

        # ---- (B) 各 BS 独立做功率搜索 + RB 分配 ----
        slot_info = {'slot': k}
        for b in (1, 2, 3):
            if not assoc[b]:                    # 没用户
                slot_info.update({f'BS{b}_P': 0,
                                   f'BS{b}_QoS': 0,
                                   f'BS{b}_RB':  {}})
                continue
            row      = df_bs[b].iloc[idx]
            gain_db  = {u: row[u] for u in assoc[b]}
            p_opt, alloc_opt, q_opt = best_power(assoc[b], gain_db)

            slot_info.update({f'BS{b}_P':   p_opt,
                               f'BS{b}_QoS': q_opt,
                               f'BS{b}_RB':  alloc_opt})
        results.append(slot_info)

    # 打印首个时隙方案示例
    print(results[0])

if __name__ == '__main__':
    main()
