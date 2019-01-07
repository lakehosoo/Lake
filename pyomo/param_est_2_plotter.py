from pyomo.environ import *
from pyomo.dae import *

a_conc = {0.1:0.606, 0.2:0.368, 0.3:0.223, 0.4:0.135, 0.5:0.082,
          0.6:0.05, 0.7:0.03, 0.8:0.018, 0.9:0.011, 1.0:0.007}

b_conc = {0.1:0.373, 0.2:0.564, 0.3:0.647, 0.4:0.669, 0.5:0.656,
          0.6:0.624, 0.7:0.583, 0.8:0.539, 0.9:0.494, 1.0:0.451}

m = ConcreteModel()

m.time_meas = Set(initialize=sorted(a_conc.keys()),ordered=True)
m.a_meas = Param(m.time_meas, initialize=a_conc)
m.b_meas = Param(m.time_meas, initialize=b_conc)

m.time = ContinuousSet(initialize=m.time_meas, bounds=(0,1))
m.a = Var(m.time)
m.b = Var(m.time)
m.c = Var(m.time)
m.k1 = Var(bounds=(0,10))
m.k2 = Var(bounds=(0,10))

m.dadt = DerivativeVar(m.a, wrt=m.time)
m.dbdt = DerivativeVar(m.b, wrt=m.time)
m.dcdt = DerivativeVar(m.c, wrt=m.time)

def _init_conditions(m):
    yield m.a[0] == 1
    yield m.b[0] == 0
m.init_conditions= ConstraintList(rule=_init_conditions)

def _dadt(m,i):
    return m.dadt[i] == -m.k1*m.a[i]
m.dadtcon = Constraint(m.time, rule=_dadt)

def _dbdt(m,i):
    return m.dbdt[i] == m.k1*m.a[i] - m.k2*m.b[i]
m.dbdtcon = Constraint(m.time, rule=_dbdt)

def _dcdt(m,i):
    return m.dcdt[i] == m.k2*m.b[i]
m.dcdtcon = Constraint(m.time, rule=_dcdt)

def _obj(m):
    return sum((m.a[i]-m.a_meas[i])**2 + (m.b[i]-m.b_meas[i])**2 for i in m.time_meas)
m.obj = Objective(rule=_obj)

# Discretize model using Orthogonal Collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=10,ncp=5)
results = SolverFactory('ipopt').solve(m, tee=True)

def plotter(numplot, x,*y,**kwds):
    plt.subplot(numplot)
    for i,_y in enumerate(y):
        plt.plot(list(x), [value(_y[t]) for t in x],'brgcmk'[i%6])
        if kwds.get('points', False):
            plt.plot(list(x), [value(_y[t]) for t in x], 'o')
    plt.title(kwds.get('title',''))
    plt.legend(tuple(_y.name for _y in y))
    plt.xlabel(x.name)
                 
import matplotlib.pyplot as plt
plotter(111, m.time, m.a, m.b, m.c, title='Dynamic Parameter Estimation Using Collocation')

t_meas = list(m.time_meas)
a_meas = [value(m.a_meas[i]) for i in t_meas]
b_meas = [value(m.b_meas[i]) for i in t_meas]
plt.plot(t_meas,a_meas,'bo')
plt.plot(t_meas,b_meas,'ro')

print('k1 =', value(m.k1))
print('k2 =', value(m.k2))
