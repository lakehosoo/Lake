from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.t = ContinuousSet(bounds=(0,10))

m.c1 = Param(initialize=0.13)
m.c2 = Param(initialize=0.20)
m.Ac = Param(initialize=2) # Unit: m^2
m.qin1 = Param(initialize=0.5) # Unit: m^3/hr

m.qout1 = Var(m.t, initialize =0, within=NonNegativeReals)
m.qout2 = Var(m.t, initialize =0, within=NonNegativeReals)
m.overflow1 = Var(m.t, initialize =0, within=NonNegativeReals)
m.overflow2 = Var(m.t, initialize =0, within=NonNegativeReals)

m.h1 = Var(m.t, bounds=(0,1))
m.h2 = Var(m.t, bounds=(0,1))
m.dh1dt = DerivativeVar(m.h1, wrt=m.t)
m.dh2dt = DerivativeVar(m.h2, wrt=m.t)
m.h1[0].fix(0)
m.h2[0].fix(0)

def _dh1dt(m,i):
    return m.Ac*m.dh1dt[i] == m.qin1 - m.qout1[i] - m.overflow1[i]
m.dh1dtcon = Constraint(m.t, rule=_dh1dt)

def _dh2dt(m,i):
    return m.Ac*m.dh2dt[i] == m.qout1[i] - m.qout2[i] - m.overflow2[i]
m.dh2dtcon = Constraint(m.t, rule=_dh2dt)

def _qout1(m,i):
    return (m.qout1[i]/m.c1)**2 == m.h1[i]
m.qout1con = Constraint(m.t, rule=_qout1)

def _qout2(m,i):
    return (m.qout2[i]/m.c1)**2 == m.h2[i]
m.qout2con = Constraint(m.t, rule=_qout2)

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=10,ncp=7,scheme='LAGRANGE-RADAU')

def _obj(m):
    return sum(m.overflow1[i] + m.overflow2[i] for i in m.t)
m.obj = Objective(rule=_obj)

results = SolverFactory('ipopt').solve(m, tee=True)
'''
# plot results
plt.figure()
plt.plot(t,y[:,0],'b-')
plt.plot(t,y[:,1],'r--')
plt.xlabel('Time (hrs)')
plt.ylabel('Height (m)')
plt.legend(['h1','h2'])
plt.show()
'''
