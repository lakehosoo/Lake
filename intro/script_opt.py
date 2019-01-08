from pyomo.environ import*
import numpy as np

m = ConcreteModel()

m.x1 = Var(initialize=1, bounds=(1,5))
m.x2 = Var(initialize=5, bounds=(1,5))
m.x3 = Var(initialize=5, bounds=(1,5))
m.x4 = Var(initialize=1, bounds=(1,5))

def _constraint1(m):
    return 25 - m.x1*m.x2*m.x3*m.x4 <= 0
m.constraint1 = Constraint(rule=_constraint1)

def _constraint2(m):
    return m.x1**2 + m.x2**2 + m.x3**2 + m.x4**2 == 40
m.constraint2 = Constraint(rule=_constraint2)

def _objective(m):
    return m.x1*m.x4*(m.x1 + m.x2 + m.x3) + m.x3
m.objective = Objective(rule=_objective)

result = SolverFactory('ipopt').solve(m)

print('Objective Value: ' + str(value(m.objective)))
print('Solution')
print('x1 = ' + str(value(m.x1)))
print('x2 = ' + str(value(m.x2)))
print('x3 = ' + str(value(m.x3)))
print('x4 = ' + str(value(m.x4)))
