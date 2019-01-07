from pyomo.environ import *
import pandas as pd

# load the data file
data_file = pd.read_csv('data_regression/data.csv')

xm = data_file['xm']
ym = data_file['ym']
xm_dict = xm.to_dict()
ym_dict = ym.to_dict()

m = ConcreteModel()

m.N = Set(initialize=range(len(xm)))
m.x_meas = Param(m.N, initialize=xm_dict)
m.y_meas = Param(m.N, initialize=ym_dict)

m.a = Var()
m.b = Var()
m.c = Var()
m.y_est = Var(m.N)

def _estimation(m,i):
    return m.y_est[i] == m.a + m.b/m.x_meas[i] + m.c*log(m.x_meas[i])
m.estimation = Constraint(m.N, rule=_estimation)

def _obj(m):
    return sum((m.y_est[i]-m.y_meas[i])**2 for i in m.N)
m.obj = Objective(rule=_obj)

results = SolverFactory('ipopt').solve(m, tee=True)
#SolverFactory('ipopt').solve(m, tee=True)
#m.pprint()

# print optimized parameters
print('Fitted a = ' + str(value(m.a)))
print('Fitted b = ' + str(value(m.b)))
print('Fitted c = ' + str(value(m.c)))

N = list(m.N)
x_meas = [value(m.x_meas[i]) for i in N]
y_meas = [value(m.y_meas[i]) for i in N]
y_est  = [value(m.y_est[i])  for i in N]

# plot data
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_meas,y_meas,'ro')
plt.plot(x_meas,y_est,'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Measured','Predicted'], loc='best')
plt.show()
