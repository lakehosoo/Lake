# Describing the column dynamics for delt
# given input and initial concentration, this code gives one step after states for the case of column and dead volume

# Danckwart boundary condition is used with some assumption
# Column dynamics with mass transfer resistance and axial dispersion
# Modified langmuir isotherm is used
# Temperature terms are added
# 2 component only
# PDEs are solved by method of line

def column(delt,c0,c,T0,eb,gamma,v,D,k,qm,K,E,N,delz):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Defining the dynamics
    def f_rhs(t,y,e,gamma,v,D,k,qm,K,E,N,delz,ca0,cb0,T0):
        R = 8.3145
        out = [0]*(5*N)
        # Boundary condition
        out[0*N] = (v**2/D)*(ca0 - y[0]) + D*(y[1] - y[0])/(delz**2)
        out[1*N-1] = D*(y[N-1] - 2*y[N-1] + y[N-2])/(delz**2) - v*(y[N-1] - y[N-2])/(delz) - ((1-e)/e)*k[0]*(qm[0]*K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]) - y[2*N-1])
        out[1*N] = 0
        out[2*N-1] = k[0]*(qm[0]*K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]) - y[2*N-1])
        out[2*N] = (v**2/D)*(cb0 - y[2*N]) + D*(y[2*N + 1] - y[2*N])/(delz**2)
        out[3*N-1] = D*(y[3*N-1] - 2*y[3*N-1] + y[3*N-2])/(delz**2) - v*(y[3*N-1] - y[3*N-2])/(delz) - ((1-e)/e)*k[1]*(qm[1]*K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]) - y[4*N-1])
        out[3*N] = 0
        out[4*N-1] = k[1]*(qm[1]*K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[N-1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[3*N-1]) - y[4*N-1])
        out[4*N] = (gamma*v**2/D)*(T0 - y[4*N]) + gamma*D*(y[4*N+1] - y[4*N])/(delz**2)
        out[5*N-1] = gamma*D*(y[5*N-1] - 2*y[5*N-1] + y[5*N-2])/(delz**2) - gamma*v*(y[5*N-1] - y[5*N-2])/(delz)

        # Internal dynamics
        for i in range(N-2):
            out[0*N+i+1] = D*(y[i+2] - 2*y[i+1] + y[i])/(delz**2) - v*(y[i+1] - y[i])/(delz) - ((1-e)/e)*k[0]*(qm[0]*K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[i+1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[i+1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[2*N+i+1]) - y[N+i+1])
            out[1*N+i+1] = k[0]*(qm[0]*K[0]*np.exp(E[0]/(R*y[4*N+i+1]))*y[i+1]/(1+K[0]*np.exp(E[0]/(R*y[4*N+i+1]))*y[i+1]+K[1]*np.exp(E[1]/(R*y[4*N+i+1]))*y[2*N+i+1]) - y[N+i+1])
            out[2*N+i+1] = D*(y[2*N+i+2] - 2*y[2*N+i+1] + y[2*N+i])/(delz**2) - v*(y[2*N+i+1] - y[2*N+i])/(delz) - ((1-e)/e)*k[1]*(qm[1]*K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[2*N+i+1]/(1+K[0]*np.exp(E[0]/(R*y[5*N-1]))*y[i+1]+K[1]*np.exp(E[1]/(R*y[5*N-1]))*y[2*N+i+1]) - y[3*N+i+1])
            out[3*N+i+1] = k[1]*(qm[1]*K[1]*np.exp(E[1]/(R*y[4*N+i+1]))*y[2*N+i+1]/(1+K[0]*np.exp(E[0]/(R*y[4*N+i+1]))*y[i+1]+K[1]*np.exp(E[1]/(R*y[4*N+i+1]))*y[2*N+i+1]) - y[3*N+i+1])
            out[4*N+i+1] = gamma*D*(y[4*N+i+2] - 2*y[4*N+i+1] + y[4*N+i])/(delz**2) - gamma*v*(y[4*N+i+1] - y[4*N+i])/(delz)
        return out

    fun = lambda t, x: f_rhs(t,x,eb,gamma,v,D,k,qm,K,E,N,delz,c0[0],c0[1],T0)

    # Simulation Part

    # Initialization

    # Solving the dynamics
    sol = solve_ivp(fun, [0, delt], c, t_eval=[delt], vectorized=True)
    slice_c = np.c_[c,sol.y]
    next_c = slice_c[:,-1]
    c_out = [next_c[N-1], next_c[3*N-1]]
    T_out = next_c[5*N-1]

    return c_out, T_out, next_c

def columnd(delt,c0,c,T0,eb,gamma,v,D,k,qm,K,E,Nex,delzd):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Defining the dynamics
    def f_rhsd(t,y,e,gamma,v,D,k,qm,K,E,Nex,delzd,ca0,cb0,T0):
        R = 8.3145
        out = [0]*(3*Nex)
        # Boundary condition
        out[0*Nex] = (v**2/D)*(ca0 - y[0]) + D*(y[1] - y[0])/(delzd**2)
        out[1*Nex-1] = D*(y[1*Nex-1] - 2*y[1*Nex-1] + y[1*Nex-2])/(delzd**2) - v*(y[1*Nex-1] - y[1*Nex-2])/(delzd)
        out[1*Nex] = (v**2/D)*(cb0 - y[1*Nex]) + D*(y[1*Nex+1] - y[1*Nex])/(delzd**2)
        out[2*Nex-1] = D*(y[2*Nex-1] - 2*y[2*Nex-1] + y[2*Nex-2])/(delzd**2) - v*(y[2*Nex-1] - y[2*Nex-2])/(delzd)
        out[2*Nex] = (gamma*v**2/D)*(T0 - y[2*Nex]) + gamma*D*(y[2*Nex+1] - y[2*Nex])/(delzd**2)
        out[3*Nex-1] = gamma*D*(y[3*Nex-1] - 2*y[3*Nex-1] + y[3*Nex-2])/(delzd**2) - gamma*v*(y[3*Nex-1] - y[3*Nex-2])/(delzd)

        # Internal dynamics
        for i in range(Nex-2):
            out[i+1] = D*(y[i+2] - 2*y[i+1] + y[i])/(delzd**2) - v*(y[i+1] - y[i])/(delzd)
            out[1*Nex+i+1] = D*(y[1*Nex+i+2] - 2*y[1*Nex+i+1] + y[1*Nex+i])/(delzd**2) - v*(y[1*Nex+i+1] - y[1*Nex+i])/(delzd)
            out[2*Nex+i+1] = gamma*D*(y[2*Nex+i+2] - 2*y[2*Nex+i+1] + y[2*Nex+i])/(delzd**2) - gamma*v*(y[2*Nex+i+1] - y[2*Nex+i])/(delzd)

        return out

    fund = lambda t, x: f_rhsd(t,x,eb,gamma,v,D,k,qm,K,E,Nex,delzd,c0[0],c0[1],T0)

    # Simulation Part

    # Initialization

    # Solving the dynamics
    sol = solve_ivp(fund, [0, delt], c, t_eval=[delt], vectorized=True)
    slice_c = np.c_[c,sol.y]
    next_c = slice_c[:,-1]
    c_out = [next_c[Nex-1], next_c[2*Nex-1]]
    T_out = next_c[3*Nex-1]

    return c_out, T_out, next_c
