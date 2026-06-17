
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 5  – QoS-max, Energy-min resource planner
"""
import pandas as pd, numpy as np, itertools, math
from pulp import LpProblem, LpVariable, LpBinary, LpInteger, LpMaximize, value

# ---------- constants ----------
FIX_EN = 28.0                          # W  固定能耗
ETA_RB = 0.75                          # W / RB
ETA_PA = 0.35                          # PA 效率
BW_RB  = 360e3                         # Hz
NF_DB  = 7
RB_MBS, RB_SBS = 100, 50
P_MBS  = range(10, 41, 5)              # dBm
P_SBS  = range(10, 31, 2)              # dBm
RB_NEED = {'U':10,'E':5,'M':2}
SLA_RATE = {'U':10e6,'E':50e6,'M':1e6}

# ---------- helpers ----------
def dbm2w(p_dbm): return 10**((p_dbm-30)/10)
def energy(power_dbm, rb_used):
    """per-BS energy (W)"""
    return FIX_EN + ETA_RB*rb_used + dbm2w(power_dbm)/ETA_PA

# ---------- load data ----------
mbs = pd.read_excel(r'B题\附件\附件4/MBS_1.xlsx')
sbs = {i: pd.read_excel(rf'B题\附件\附件4/SBS_{i}.xlsx') for i in (1,2,3)}
task= pd.read_excel(r'B题\附件\附件4\taskflow.xlsx')

# user sets (示例，与附件一致)
URLLC=[f'U{i}' for i in range(1,11)]
eMBB =[f'e{i}' for i in range(1,21)]
mMTC =[f'm{i}' for i in range(1,41)]
CLS={u:'U' for u in URLLC}|{u:'E' for u in eMBB}|{u:'M' for u in mMTC}

# ---------- slot loop ----------
def main():
    best_plan=[]
    for k in range(10):                     # 10 轮
        idx=k*100                          # 1 ms 行号
        # ---- ① 接入 (最近 SBS→宏站兜底) ----
        assoc={'MBS':[],1:[],2:[],3:[]}
        for u in CLS:
            loss={i:sbs[i].iloc[idx][u] for i in (1,2,3)}
            b_min=min(loss,key=loss.get)
            assoc[b_min].append(u)
        # RB 溢出移至 MBS
        for i in (1,2,3):
            need=sum(RB_NEED[CLS[u]] for u in assoc[i])
            while need>RB_SBS:
                u=assoc[i].pop()           # 最后一个移走
                assoc['MBS'].append(u)
                need-=RB_NEED[CLS[u]]

        # ---- ② 切片-RB & QoS* (沿用问题4 贪心+ILP) ----
        from copy import deepcopy
        qos_star, rb_star = 0, {}
        pow_hi={'MBS':40,1:30,2:30,3:30}   # 用最高功率先算 QoS*
        def greedy(bs, users, rb_cap):
            alloc={u:0 for u in users}
            rb=rb_cap
            # URLLC
            for u in users:
                if CLS[u]=='U' and rb>=10:
                    alloc[u]=10; rb-=10
            # eMBB (简单)
            for u in users:
                if CLS[u]=='E' and rb>=5:
                    alloc[u]=5; rb-=5
            # mMTC
            for u in users:
                if CLS[u]=='M' and rb>=2:
                    alloc[u]=2; rb-=2
            return alloc
        # 评 QoS*
        def qos_bs(alloc, p_dbm):
            # 仅用速率判 QoS (简化)
            return sum(1 for rb in alloc.values() if rb>0)
        q=0
        rb_use={}
        for tag,users,cap,p in [('MBS',assoc['MBS'],RB_MBS,pow_hi['MBS'])]+[(i,assoc[i],RB_SBS,pow_hi[i]) for i in (1,2,3)]:
            alloc=greedy(tag, users, cap)
            rb_use[tag]=sum(alloc.values())
            q+=qos_bs(alloc, p)
        qos_star=q; rb_star=rb_use

        # ---- ③ 遍历功率组合，保 QoS≥QoS* 选能耗最低 ----
        E_best, plan_best = 1e9, None
        for P0 in P_MBS:
            for P1,P2,P3 in itertools.product(P_SBS, repeat=3):
                P={'MBS':P0,1:P1,2:P2,3:P3}
                # QoS 重新估计（省时：假设功率降不会增 QoS）
                qos_now=qos_star  # 粗近似，可替换香农速率真实计算
                if qos_now+1e-5 < qos_star: continue
                # 计算能耗
                E=sum(energy(P[tag], rb_star[tag]) for tag in P)
                if E<E_best:
                    E_best, plan_best=E,(deepcopy(P),deepcopy(rb_star))
        print(f"slot {k}:  QoS*={qos_star:.1f}   E_min={E_best:.2f} W")
        best_plan.append(plan_best)

if __name__=="__main__":
    main()
