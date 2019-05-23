# Describing the column dynamics for delta t
# given input and initial concentration, this code gives one step after states for the case of column 

# EDM with Method of line

import numpy as np
from scipy.integrate import solve_ivp

def column(c0, c, e, v, H, N, delt, delz):

    # Defining the dynamics
    def f_rhs(t, y, e, v, H, N, delz):
        
        out = np.zeros(2*N)
        
        # Boundary condition
        out[0*N] = 0
        out[1*N] = 0

        # Internal dynamics
        for i in range(N-1):
            out[0*N+i+1] = - v*(y[0*N+i+1] - y[0*N+i])/((1 + H[0]*(1-e)/e)*(delz))
            out[1*N+i+1] = - v*(y[1*N+i+1] - y[1*N+i])/((1 + H[1]*(1-e)/e)*(delz))

        return out

    fun = lambda t, x: f_rhs(t, x, e, v, H, N, delz)

    # Initial condition seeting
    c[0*N], c[1*N] = c0[0], c0[1]
    
    # Solving the dynamics
    sol = solve_ivp(fun, [0, delt], c, method='RK45', t_eval=[delt], vectorized=True)
    next_c = sol.y[:,-1]
    c_out = np.array([next_c[1*N-1], next_c[2*N-1]])

    return c_out, next_c
