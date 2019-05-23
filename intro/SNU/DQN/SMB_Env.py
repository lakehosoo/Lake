# SMB Environment for DQN
# Accept the state and action, gives next state and reward
# Use sub function column

# States: concentration of solute A, B through column 1-8, previous action, time step, mode, tank
import numpy as np
from scipy.integrate import solve_ivp
from gym.utils import seeding


class SMB_Env(object):

    # Simulation Conditions
    Nz = 50; N = Nz
    Nt = 12

    # Basic Paramaters
    L  = 1.          # m
    D  = 0.1         # m
    e = 0.66
    H  = np.array([2, 1])
    st = 120         # sec
    cf = np.array([1,1])

    # initial value of action ( section wise )
    v = np.array([0.0191, 0.0191, 0.0135, 0.0135, 0.0165, 0.0165, 0.0105, 0.0105])
    scale_a = 50

    # Additional Values
    A = 3.141592*D*D/4
    V = A*L
    delz = L/Nz
    delt = st/Nt

    def __init__(self):

        self.s_dim = 2*8 * self.Nz + 1 + 1 + 8 + 4  # fix concentration, time step, mode, previous_action, E_tank, R_tank
        self.a_dim = 8

        self.x0 = np.array([np.hstack((np.zeros(self.s_dim - 14), 0, 0, self.scale_a*0.0191, self.scale_a*0.0191, self.scale_a*0.0135, self.scale_a*0.0135, self.scale_a*0.0165, self.scale_a*0.0165,
                                       self.scale_a*0.0105, self.scale_a*0.0105, 10 ** -1, 0, 0, 10 ** -1))])
        self.u0 = np.array([np.zeros(self.a_dim)])

        self.ts = self.delt
        self.Total_simulation_time = 7*8*self.st  # Cycle, switch, switch time

        # Reward parameter
        self.Product = 1
        self.Consum_D = 0.1
        self.Purity_E = -2
        self.Purity_R = -2

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self.x0, self.u0

    def step(self, state, act):
        # action is by section, it is delta action

        time_step = state[16 * self.N]
        mode = int(round(state[16 * self.N + 1] * 7))
        action_prev = state[16 * self.N + 2:16 * self.N + 10] / self.scale_a
        action = act
        E_tank = state[16 * self.N + 10: 16 * self.N + 12]
        R_tank = state[16 * self.N + 12: 16 * self.N + 14]

        role_ind = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        role_ind = np.concatenate((role_ind[8 - mode:8], role_ind[0:8 - mode]))

        s1, s2, s3, s4 = state[0 * self.N: 2 * self.N], state[ 2 * self.N: 4 * self.N], state[ 4 * self.N: 6 * self.N], state[ 6 * self.N: 8 * self.N]
        s5, s6, s7, s8 = state[8 * self.N:10 * self.N], state[10 * self.N:12 * self.N], state[12 * self.N:14 * self.N], state[14 * self.N:16 * self.N]

        s1_in = np.array([s8[self.N - 1], s8[2 * self.N - 1]])
        s2_in = np.array([s1[self.N - 1], s1[2 * self.N - 1]])
        s3_in = np.array([s2[self.N - 1], s2[2 * self.N - 1]])
        s4_in = np.array([s3[self.N - 1], s3[2 * self.N - 1]])
        s5_in = np.array([s4[self.N - 1], s4[2 * self.N - 1]])
        s6_in = np.array([s5[self.N - 1], s5[2 * self.N - 1]])
        s7_in = np.array([s6[self.N - 1], s6[2 * self.N - 1]])
        s8_in = np.array([s7[self.N - 1], s7[2 * self.N - 1]])

        c_in = np.array([s1_in, s2_in, s3_in, s4_in, s5_in, s6_in, s7_in, s8_in])

        c_in[np.where(role_ind == 0)] = c_in[np.where(role_ind == 0)] * action[7] / action[0]
        c_in[np.where(role_ind == 1)] = c_in[np.where(role_ind == 1)] * action[0] / action[1]
        c_in[np.where(role_ind == 2)] = c_in[np.where(role_ind == 2)]
        c_in[np.where(role_ind == 3)] = c_in[np.where(role_ind == 3)] * action[2] / action[3]
        c_in[np.where(role_ind == 4)] = (c_in[np.where(role_ind == 4)] * action[3] + self.cf * (action[4] - action[3])) / action[4]
        c_in[np.where(role_ind == 5)] = c_in[np.where(role_ind == 5)] * action[4] / action[5]
        c_in[np.where(role_ind == 6)] = c_in[np.where(role_ind == 6)]
        c_in[np.where(role_ind == 7)] = c_in[np.where(role_ind == 7)] * action[6] / action[7]

        [s1_out, next_c1] = self.column(s1, c_in[0], action[role_ind[0]])
        [s2_out, next_c2] = self.column(s2, c_in[1], action[role_ind[1]])
        [s3_out, next_c3] = self.column(s3, c_in[2], action[role_ind[2]])
        [s4_out, next_c4] = self.column(s4, c_in[3], action[role_ind[3]])
        [s5_out, next_c5] = self.column(s5, c_in[4], action[role_ind[4]])
        [s6_out, next_c6] = self.column(s6, c_in[5], action[role_ind[5]])
        [s7_out, next_c7] = self.column(s7, c_in[6], action[role_ind[6]])
        [s8_out, next_c8] = self.column(s8, c_in[7], action[role_ind[7]])

        s_out = np.array([s1_out, s2_out, s3_out, s4_out, s5_out, s6_out, s7_out, s8_out])
        E_tank += s_out[np.where(role_ind == 1)][0];
        puri_e = E_tank[0] / sum(E_tank)
        R_tank += s_out[np.where(role_ind == 5)][0];
        puri_r = R_tank[1] / sum(R_tank)

        if puri_e > 0.995:
            Reward_e = (puri_e - 0.995)
        elif puri_e < 0.99:
            Reward_e = 0.99 - puri_e
        else:
            Reward_e = 0

        if puri_r > 0.995:
            Reward_r = (puri_r - 0.995)
        elif puri_r < 0.99:
            Reward_r = 0.99 - puri_r
        else:
            Reward_r = 0

        time_step += 0.01

        if np.mod(int(round(100 * time_step)), self.Nt) == 0:
            mode += 1

        if mode == 8:
            mode = 0
        # + self.Consum_D * (action[1] - action[6] + action[3] - action[2] + action[5] - action[4])
        Reward = self.Product * (action[4] - action[3]) + Reward_e * self.Purity_E + Reward_r * self.Purity_R
        next_s = np.concatenate((next_c1, next_c2, next_c3, next_c4, next_c5, next_c6, next_c7, next_c8, [round(time_step, 2)], [1/7*mode], self.scale_a*action, E_tank, R_tank))

        return next_s, Reward


    def column(self, c, c0, v):
        # Defining the dynamics
        def f_rhs(t, y, u):
            out = np.zeros(2 * self.N)

            # Boundary condition
            out[0 * self.N] = 0
            out[1 * self.N] = 0

            # Internal dynamics
            for i in range(self.N - 1):
                out[0 * self.N + i + 1] = - u * (y[0 * self.N + i + 1] - y[0 * self.N + i]) / ((1 + self.H[0] * (1 - self.e) / self.e) * (self.delz))
                out[1 * self.N + i + 1] = - u * (y[1 * self.N + i + 1] - y[1 * self.N + i]) / ((1 + self.H[1] * (1 - self.e) / self.e) * (self.delz))

            return out

        fun = lambda t, x: f_rhs(t, x, v)

        # Initial condition seeting
        c[0 * self.N], c[1 * self.N] = c0[0], c0[1]

        # Solving the dynamics
        sol = solve_ivp(fun, [0, self.delt], c, method='RK45', t_eval=[self.delt], vectorized=True)
        next_c = sol.y[:, -1]
        c_out = np.array([next_c[1 * self.N - 1], next_c[2 * self.N - 1]])

        return c_out, next_c
