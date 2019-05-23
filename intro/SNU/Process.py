# Describing the SMB process
# 31/12/18 - xx/01/18 Tae Hoon Oh

# Danckwart boundary condition is used with some assumption
# Column dynamics with mass transfer resistance and axial dispersion
# Modified langmuir isotherm is used
# Temperature terms are added
# 2 component only
# PDEs are solved by method of line

import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameter Values
# Subscript for Column
# column parameters
L = 1.          # m
Ldead = 0.1     # m
D = 0.1         # m
eb = 0.66

# particle parameters
R = 8.3145                    # J/mol,K
dp = 0.56e-3                  # m
mu = 0.3e-3                   # kg/m.s = 1000 cp
sp = 0.3                      #
K = [1.053, 0.3151]           # m3/kg
qm = [0.1379, 0.1379]         # kg/m3
k = [3.335/60, 3.147/60]      # 1/sec
E = [2500, 2500]              # J/mol
gamma = 1                     #

# Calculation
A = 3.141592*D*D/4
V = A*L

# Operating Condition
ts = 120.       # sec
cf = np.array([1.,1.])      # kg/m3
TF = 298        # K
TD = 298        # K
Pe = 1000

m1 = 1.0
m2 = 0.2
m3 = 0.8
m4 = 0.1

# Calculation
Q1 = (m1*(1-eb)*V + eb*V)/ts    # m3/s
Q2 = (m2*(1-eb)*V + eb*V)/ts    # m3/s
Q3 = (m3*(1-eb)*V + eb*V)/ts    # m3/s
Q4 = (m4*(1-eb)*V + eb*V)/ts    # m3/s

QE = Q1 - Q2                    # m3/s
QF = Q3 - Q2                    # m3/s
QR = Q3 - Q4                    # m3/s
QD = Q1 - Q4                    # m3/s

v1 = Q1/(eb*A)                  # m/s
v2 = Q2/(eb*A)                  # m/s
v3 = Q3/(eb*A)                  # m/s
v4 = Q4/(eb*A)                  # m/s

Dax1 = v1*L/Pe                  # m2/s
Dax2 = v2*L/Pe                  # m2/s
Dax3 = v3*L/Pe                  # m2/s
Dax4 = v4*L/Pe                  # m2/s

Q = np.array([Q1,Q2,Q3,Q4])     # m3/s
v = np.array([v1,v2,v3,v4])     # m/s
Dax = np.array([Dax1,Dax2,Dax3,Dax4])   # m2/s

delp = L*(180*mu*(1-eb)**2*eb*v)/(10e5*(sp**2)*(dp**2))     # bar

# Simulation Condition
Nz = 80  #180     # for a column
Nd = 40  #100   # for a dead volume
Nt = 12

# Calculation
N = Nz + 1
Nex = Nd + 1
delz = L/Nz
delzd = Ldead/Nd
delt = ts/Nt

# Define sections (8 sections with 8 dead volumes)
c1, c2, c3, c4, c5, c6, c7, c8 = np.zeros(5*N), np.zeros(5*N), np.zeros(5*N), np.zeros(5*N), np.zeros(5*N), np.zeros(5*N), np.zeros(5*N), np.zeros(5*N) # ca,qa,cb,qb,T
cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8 = np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex), np.zeros(3*Nex) # ca,cb,T

# Define output
E_tank = np.array([])
R_tank = np.array([])

# Initialization

# 초기 단계의 input
c1_out, c2_out, c3_out, c4_out, c5_out, c6_out, c7_out, c8_out = [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]
cd1_out, cd2_out, cd3_out, cd4_out, cd5_out, cd6_out, cd7_out, cd8_out = [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]
T1_out, T2_out, T3_out, T4_out, T5_out, T6_out, T7_out, T8_out = 290., 290., 290., 290., 290., 290., 290., 290.
Td1_out, Td2_out, Td3_out, Td4_out, Td5_out, Td6_out, Td7_out, Td8_out = 290., 290., 290., 290., 290., 290., 290., 290.

# c 설정 (전부 0으로 시작)

# T 설정 (전부 290으로 시작)
c1[4*N:5*N], c2[4*N:5*N], c3[4*N:5*N], c4[4*N:5*N], c5[4*N:5*N], c6[4*N:5*N], c7[4*N:5*N], c8[4*N:5*N] = 290*np.ones(N), 290*np.ones(N), 290*np.ones(N), 290*np.ones(N), 290*np.ones(N), 290*np.ones(N), 290*np.ones(N), 290*np.ones(N)
cd1[2*Nex:3*Nex], cd2[2*Nex:3*Nex], cd3[2*Nex:3*Nex], cd4[2*Nex:3*Nex], cd5[2*Nex:3*Nex], cd6[2*Nex:3*Nex], cd7[2*Nex:3*Nex], cd8[2*Nex:3*Nex] = 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex), 290*np.ones(Nex)

# Simulation Part
from Column import *

#Initial logic
cindvec = np.array([cd8_out, cd1_out, cd2_out, cd3_out, cd4_out, cd5_out, cd6_out, cd7_out])        # Input logic을 위한 벡터
Tindvec = np.array([Td8_out, Td1_out, Td2_out, Td3_out, Td4_out, Td5_out, Td6_out, Td7_out])
Eind, Find, Rind, Dind = 1, 3, 5, -1        # 시작 하자 마자 포지션 바뀌기 떄문에 한 단계 전 상태를 설정 ( 들어가는 column 기준 index임 )
vind = np.array([0,1,1,2,2,3,3,0])          # 시작 하자 마자 포지션 바뀌기 떄문에 한 단계 전 상태를 설정

for kk in range(8*20): # section number * cycle number

    Eind = np.mod(Eind + 1, 8); Find = np.mod(Find + 1, 8); Rind = np.mod(Rind + 1, 8); Dind = np.mod(Dind + 1, 8)  # Feeding 위치 변경
    vind = np.hstack((vind[7],vind[0:7]))   # Section 속도 변경

    for ii in range(Nt):
        print(kk)
        print(ii)

        cindvec[Dind] = (v[vind[Dind - 1]] / v[vind[Dind]]) * cindvec[Dind]       # Desorbent input에 의한 농도 감소
        cindvec[Find] = (v[vind[Find - 1]] * cindvec[Find] + (v[vind[Find]] - v[vind[Find - 1]]) * cf) / (v[vind[Find]])     # Feed 투입에 따른 농도 변화
        Tindvec[Dind] = (v[vind[Dind- 1]]*Tindvec[Dind] + (v[vind[Dind]] - v[vind[Dind - 1]])*TD)/v[vind[Dind]]
        Tindvec[Find] = (v[vind[Find - 1]] * Tindvec[Find] + (v[vind[Find]] - v[vind[Find - 1]]) * TF) / (v[vind[Find]])
        # print(v)
        # print((v[vind[Find]] - v[vind[Find - 1]]) )
        c1in = cindvec[0]
        cd1in = c1_out
        c2in = cindvec[1]
        cd2in = c2_out
        c3in = cindvec[2]
        cd3in = c3_out
        c4in = cindvec[3]
        cd4in = c4_out
        c5in = cindvec[4]
        cd5in = c5_out
        c6in = cindvec[5]
        cd6in = c6_out
        c7in = cindvec[6]
        cd7in = c7_out
        c8in = cindvec[7]
        cd8in = c8_out

        T1in = Tindvec[0]
        Td1in = T1_out
        T2in = Tindvec[1]
        Td2in = T2_out
        T3in = Tindvec[2]
        Td3in = T3_out
        T4in = Tindvec[3]
        Td4in = T4_out
        T5in = Tindvec[4]
        Td5in = T5_out
        T6in = Tindvec[5]
        Td6in = T6_out
        T7in = Tindvec[6]
        Td7in = T7_out
        T8in = Tindvec[7]
        Td8in = T8_out

        [c1_out, T1_out, next_c1] = column(delt, c1in, c1, T1in, eb, gamma, v[vind[0]], Dax[vind[0]], k, qm, K, E, N, delz)
        [c2_out, T2_out, next_c2] = column(delt, c2in, c2, T2in, eb, gamma, v[vind[1]], Dax[vind[1]], k, qm, K, E, N, delz)
        [c3_out, T3_out, next_c3] = column(delt, c3in, c3, T3in, eb, gamma, v[vind[2]], Dax[vind[2]], k, qm, K, E, N, delz)
        [c4_out, T4_out, next_c4] = column(delt, c4in, c4, T4in, eb, gamma, v[vind[3]], Dax[vind[3]], k, qm, K, E, N, delz)
        [c5_out, T5_out, next_c5] = column(delt, c5in, c5, T5in, eb, gamma, v[vind[4]], Dax[vind[4]], k, qm, K, E, N, delz)
        [c6_out, T6_out, next_c6] = column(delt, c6in, c6, T6in, eb, gamma, v[vind[5]], Dax[vind[5]], k, qm, K, E, N, delz)
        [c7_out, T7_out, next_c7] = column(delt, c7in, c7, T7in, eb, gamma, v[vind[6]], Dax[vind[6]], k, qm, K, E, N, delz)
        [c8_out, T8_out, next_c8] = column(delt, c8in, c8, T8in, eb, gamma, v[vind[7]], Dax[vind[7]], k, qm, K, E, N, delz)

        [cd1_out, Td1_out, next_cd1] = columnd(delt, cd1in, cd1, Td1in, eb, gamma, v[vind[0]], Dax[vind[0]], k, qm, K, E, Nex, delzd)
        [cd2_out, Td2_out, next_cd2] = columnd(delt, cd2in, cd2, Td2in, eb, gamma, v[vind[1]], Dax[vind[1]], k, qm, K, E, Nex, delzd)
        [cd3_out, Td3_out, next_cd3] = columnd(delt, cd3in, cd3, Td3in, eb, gamma, v[vind[2]], Dax[vind[2]], k, qm, K, E, Nex, delzd)
        [cd4_out, Td4_out, next_cd4] = columnd(delt, cd4in, cd4, Td4in, eb, gamma, v[vind[3]], Dax[vind[3]], k, qm, K, E, Nex, delzd)
        [cd5_out, Td5_out, next_cd5] = columnd(delt, cd5in, cd5, Td5in, eb, gamma, v[vind[4]], Dax[vind[4]], k, qm, K, E, Nex, delzd)
        [cd6_out, Td6_out, next_cd6] = columnd(delt, cd6in, cd6, Td6in, eb, gamma, v[vind[5]], Dax[vind[5]], k, qm, K, E, Nex, delzd)
        [cd7_out, Td7_out, next_cd7] = columnd(delt, cd7in, cd7, Td7in, eb, gamma, v[vind[6]], Dax[vind[6]], k, qm, K, E, Nex, delzd)
        [cd8_out, Td8_out, next_cd8] = columnd(delt, cd8in, cd8, Td8in, eb, gamma, v[vind[7]], Dax[vind[7]], k, qm, K, E, Nex, delzd)

        c1, c2, c3, c4, c5, c6, c7, c8 = next_c1, next_c2, next_c3, next_c4, next_c5, next_c6, next_c7, next_c8
        cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8 = next_cd1, next_cd2, next_cd3, next_cd4, next_cd5, next_cd6, next_cd7, next_cd8

        cindvec = np.array([cd8_out, cd1_out, cd2_out, cd3_out, cd4_out, cd5_out, cd6_out, cd7_out])
        Tindvec = np.array([Td8_out, Td1_out, Td2_out, Td3_out, Td4_out, Td5_out, Td6_out, Td7_out])

        E_tank = np.hstack((E_tank,cindvec[Eind]))
        R_tank = np.hstack((R_tank,cindvec[Rind]))

# Checking & Plotting
ca_all = np.hstack((c1[0:N], cd1[0:Nex], c2[0:N], cd2[0:Nex], c3[0:N], cd3[0:Nex], c4[0:N], cd4[0:Nex], c5[0:N], cd5[0:Nex], c6[0:N], cd6[0:Nex], c7[0:N], cd7[0:Nex], c8[0:N], cd8[0:Nex]))
qa_all = np.hstack((c1[N:2*N], c2[N:2*N], c3[N:2*N], c4[N:2*N], c5[N:2*N], c6[N:2*N], c7[N:2*N], c8[N:2*N]))
cb_all = np.hstack((c1[2*N:3*N], cd1[Nex:2*Nex], c2[2*N:3*N], cd2[Nex:2*Nex], c3[2*N:3*N], cd3[Nex:2*Nex], c4[2*N:3*N], cd4[Nex:2*Nex], c5[2*N:3*N], cd5[Nex:2*Nex], c6[2*N:3*N], cd6[Nex:2*Nex], c7[2*N:3*N], cd7[Nex:2*Nex], c8[2*N:3*N], cd8[Nex:2*Nex]))
qb_all = np.hstack((c1[3*N:4*N], c2[3*N:4*N], c3[3*N:4*N], c4[3*N:4*N], c5[3*N:4*N], c6[3*N:4*N], c7[3*N:4*N], c8[3*N:4*N]))
T_all = np.hstack((c1[4*N:5*N], cd1[2*Nex:3*Nex], c2[4*N:5*N], cd2[2*Nex:3*Nex], c3[4*N:5*N], cd3[2*Nex:3*Nex], c4[4*N:5*N], cd4[2*Nex:3*Nex], c5[4*N:5*N], cd5[2*Nex:3*Nex], c6[4*N:5*N], cd6[2*Nex:3*Nex], c7[4*N:5*N], cd7[2*Nex:3*Nex], c8[4*N:5*N], cd8[2*Nex:3*Nex]))

np.savetxt('ca_all.csv', ca_all, fmt="%d", delimiter=",")
np.savetxt('qa_all.csv', qa_all, fmt="%d", delimiter=",")
np.savetxt('cb_all.csv', cb_all, fmt="%d", delimiter=",")
np.savetxt('qb_all.csv', qb_all, fmt="%d", delimiter=",")
np.savetxt('T_all.csv', T_all, fmt="%d", delimiter=",")
