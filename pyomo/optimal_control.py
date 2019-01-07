from pyomo.environ import*
from pyomo.dae import*

model = m = ConcreteModel()
m.tf = Param(initialize = 1)
m.t = ContinuousSet(bounds=(0, m.tf))

m.u = Var(m.t, initialize=0)
m.x1 = Var(m.t)
m.x2 = Var(m.t)
m.x3 = Var(m.t)

m.dx1dt = DerivativeVar(m.x1, wrt=m.t)
m.dx2dt = DerivativeVar(m.x2, wrt=m.t)
m.dx3dt = DerivativeVar(m.x3, wrt=m.t)

m.obj = Objective(expr=m.x3[m.tf])

def _x1dot(m, t):
    return m.dx1dt[t] == m.x2[t]
m.x1dot = Constraint(m.t, rule=_x1dot)

def _x2dot(m, t):
    return m.dx2dt[t] == -m.x2[t] + m.u[t]
m.x2dot = Constraint(m.t, rule=_x2dot)

def _x3dot(m, t):
    return m.dx3dt[t] == m.x1[t]**2 + m.x2[t]**2 + 0.005*m.u[t]**2
m.x3dot = Constraint(m.t, rule=_x3dot)

def _con(m, t):
    return m.x2[t] -8*(t-0.5)**2 + 0.5 <= 0
m.con= Constraint(m.t, rule=_con)

def _init(m):
    yield m.x1[0] == 0
    yield m.x2[0] == -1
    yield m.x3[0] == 0
m.init_conditions= ConstraintList(rule=_init)

# Discretize model using radau collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=7,ncp=6,scheme='LAGRANGE-RADAU')

# Control variable u made constant over each finite element
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

# Solve algebraic model
results = SolverFactory('ipopt').solve(m)

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
plotter(121, m.t, m.x1, m.x2, title='Differential Variables')
plotter(122, m.t, m.u, title='Control Variable', points=True)
plt.show()
