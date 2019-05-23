import matplotlib.pyplot as plt
import numpy as np

def state_plot(c, N, scale):

    con= c[0:16*N]
    time_step = c[16*N]*100
    mode = int(round(c[16*N+1]*7))
    action_prev = c[16*N+2:16*N+10] / scale
    E_tank = c[16*N+10:16*N+12]
    R_tank = c[16*N+12:16*N+14]
    puri_e = E_tank[0] / sum(E_tank)
    puri_r = R_tank[1] / sum(R_tank)

    conA = np.concatenate((c[0*N:1*N], c[2*N:3*N], c[4*N:5*N], c[6*N:7*N], c[8*N:9*N], c[10*N:11*N], c[12*N:13*N], c[14*N:15*N]))
    conB = np.concatenate((c[1*N:2*N], c[3*N:4*N], c[5*N:6*N], c[7*N:8*N], c[9*N:10*N], c[11*N:12*N], c[13*N:14*N], c[15*N:16*N]))
    conA = np.concatenate((conA[8*N - mode*N:8*N], conA[0*N:8*N - mode*N]))
    conB = np.concatenate((conB[8*N - mode*N:8*N], conB[0*N:8*N - mode*N]))

    t = np.arange(0., 8., 1/N)
    plt.figure(1)
    plt.plot(t, conA, t, conB); plt.show()

    print(mode, action_prev, puri_e, puri_r)
