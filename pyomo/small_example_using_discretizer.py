from pyomo.environ import*
from pyomo.dae import*

m   = ConcreteModel()
m.t = ContinuousSet(bounds=(0,1))

m.z     = Var(m.t)
m.dzdt  = DerivativeVar(m.z, wrt=m.t)

m.obj   = Objective(expr=1) # Dummy Objective

def _zdot(m, t):
    return m.dzdt[t] == m.z[t]**2 -2*m.z[t] + 1
m.zdot = Constraint(m.t, rule=_zdot)

def _init_con(m):
    return m.z[0] == -3
m.init_con = Constraint(rule=_init_con)

# Discretize model using backward finite difference
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m,nfe=10,scheme='BACKWARD')

# Discretize model using radau collocation
#discretizer = TransformationFactory('dae.collocation')
#discretizer.apply_to(m,nfe=1,ncp=3,scheme='LAGRANGE-RADAU')

solver = SolverFactory('ipopt')
solver.solve(m,tee=True)

import matplotlib.pyplot as plt

analytical_t = [0.01*i for i in range(0,101)]
analytical_z = [(4*t-3)/(4*t+1) for t in analytical_t]
plt.plot(analytical_t,analytical_z,'b',label='analytical solution')

time=list(m.t)
z=[value(m.z[k]) for k in m.t]
plt.plot(time,z,'ro--',label='finite difference solution')
plt.legend(loc='best')
plt.xlabel('t')
plt.show()
