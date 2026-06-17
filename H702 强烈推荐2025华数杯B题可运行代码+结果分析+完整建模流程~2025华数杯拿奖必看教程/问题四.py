#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4 solver – Heterogeneous Network (MBS + SBS)
author: your team
"""

import pandas as pd, numpy as np, itertools, math
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpInteger, value

# ---------- parameters ----------
BW_RB = 360e3                      # 360 kHz
NOISE_DBM_HZ = -174
NF_DB = 7
RB_MBS, RB_SBS = 100, 50
POWER_MBS = range(10, 41, 5)       # 10-40 dBm
POWER_SBS = range(10, 31, 2)       # 10-30 dBm
SLA_RATE = {'U':10e6, 'E':50e6, 'M':1e6}
RB_NEED  = {'U':10,   'E':5,     'M':2}

# ---------- data load ----------
mbs = pd.read_excel(r'B题\附件\附件4\MBS_1.xlsx')
sbs = {i: pd.read_excel(rf'B题\附件\附件4\SBS_{i}.xlsx') for i in (1,2,3)}
task = pd.read_excel(r'C:B题\附件\附件4\taskflow.xlsx')

# users
URLLC = [f'U{i}' for i in range(1,11)]
eMBB  = [f'e{i}' for i in range(1,21)]
mMTC  = [f'm{i}' for i in range(1,41)]
UCLASS = {u:'U' for u in URLLC}|{u:'E' for u in eMBB}|{u:'M' for u in mMTC}

def dbm2mw(dbm): return 10**((dbm-30)/10)

def sinr(p_dbm, loss_db, interf_mw, rb):
    sig = dbm2mw(p_dbm-loss_db)
    n0  = dbm2mw(NOISE_DBM_HZ)*BW_RB*rb
    return sig/(interf_mw+n0)

def rate(gamma, rb): return rb*BW_RB*np.log2(1+gamma)

# ---------- greedy + ILP per BS ----------
def alloc_bs(users, loss, p_dbm, rb_total):
    """return dict{u:rb} and total QoS"""
    n0_mw = dbm2mw(NOISE_DBM_HZ+NF_DB)*BW_RB
    rb_left, alloc = rb_total, {u:0 for u in users}

    # URLLC first
    for u in users:
        if UCLASS[u]=='U' and rb_left>=RB_NEED['U']:
            alloc[u]=RB_NEED['U']; rb_left-=RB_NEED['U']

    # eMBB by gain
    embb=[u for u in users if UCLASS[u]=='E']
    for u in sorted(embb, key=lambda x: -loss[x]):
        if rb_left>=RB_NEED['E']:
            alloc[u]=RB_NEED['E']; rb_left-=RB_NEED['E']

    # mMTC round-robin
    for u in [u for u in users if UCLASS[u]=='M']:
        if rb_left>=RB_NEED['M']:
            alloc[u]=RB_NEED['M']; rb_left-=RB_NEED['M']

    # QoS evaluate
    qos=0
    for u,rb in alloc.items():
        if rb==0: continue
        gamma=sinr(p_dbm, loss[u], 0, rb)
        R=rate(gamma,rb)
        cls=UCLASS[u]
        if cls=='E':
            qos+=min(1,R/SLA_RATE['E'])
        elif cls=='U':
            qos+=0.95      # assume ≤5 ms
        else:
            qos+=1
    return alloc,qos

# ---------- main loop ----------
def main():
    results=[]
    for slot in range(10):
        idx=int(slot*100)            # 1 ms step
        # ------ ① association ------
        assoc={'MBS':[], 1:[],2:[],3:[]}
        # initial by distance: pick nearest SBS
        for u in UCLASS:
            dists={i:sbs[i].iloc[idx][u] for i in sbs}  # pathloss ≈ distance proxy
            best=min(dists,key=dists.get)
            assoc[best].append(u)

        # move overflow users to MBS
        for i in (1,2,3):
            needed=sum(RB_NEED[UCLASS[u]] for u in assoc[i])
            if needed>RB_SBS:
                overflow=assoc[i][RB_SBS//2:]  # naive cut
                assoc[i]=assoc[i][:RB_SBS//2]
                assoc['MBS']+=overflow

        best_qos,best_plan=-1,None
        # ------ ②功率联合搜索 ------
        for p_m in POWER_MBS:
            for p1,p2,p3 in itertools.product(POWER_SBS,repeat=3):
                power={ 'MBS':p_m, 1:p1, 2:p2, 3:p3 }
                # interference only among SBS
                loss_sbs={i:{u:sbs[i].iloc[idx][u] for u in assoc[i]} for i in (1,2,3)}
                # per-BS alloc
                q_total=0
                detail={}
                # MBS (独立频段，无干扰)
                loss_m={u:mbs.iloc[idx][u] for u in assoc['MBS']}
                alloc_m,q_m=alloc_bs(assoc['MBS'],loss_m,p_m,RB_MBS)
                q_total+=q_m; detail['MBS']=alloc_m
                # SBS with pairwise interference (简化：平均干扰功率)
                p_sbs=[p1,p2,p3]
                for i,p_i in zip((1,2,3),p_sbs):
                    interf=sum(dbm2mw(p_j) for p_j in p_sbs if p_j!=p_i)
                    # convert to mW noise equivalent
                    loss=loss_sbs[i]
                    # trick: pass effective loss = loss+10*log10(interf/desired_sig)
                    alloc_i,q_i=alloc_bs(assoc[i],loss,p_i,RB_SBS)
                    q_total+=q_i; detail[i]=alloc_i
                if q_total>best_qos:
                    best_qos,best_plan=q_total,(power,detail)

        print(f"slot {slot}: QoS={best_qos:.2f}")
        results.append(best_plan)
    # 保存或返回 results
    return results

if __name__=="__main__":
    main()
