from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

# stock ticker symbol
url = 'http://apmonitor.com/che263/uploads/Main/goog.csv'

# import data with pandas
mydata = pd.read_csv(url)

m = ConcreteModel()
#m.N = Set(initialize=mydata.shape[0])
mydata_dict=mydata.to_dict()
m.N = Set(initialize=range(len(list(mydata_dict.items())[0][1])))
m.x_meas = Param(m.N, initialize=mydata_dict['Open'])
m.y_meas = Param(m.N, initialize=mydata_dict['Close'])

m.a = Var()
m.b = Var()
m.c = Var()
m.y_est = Var(m.N)

def _estimation(m,i):
    return m.y_est[i] == m.b*exp(m.a*m.x_meas[i]) + m.c
m.estimation = Constraint(m.N, rule=_estimation)

def _obj(m):
    return sum((m.y_est[i]-m.y_meas[i])**2 for i in m.N)
m.obj = Objective(rule=_obj)

results = SolverFactory('ipopt').solve(m, tee=True)
#SolverFactory('ipopt').solve(m, tee=True)
#m.pprint()

# print optimized parameters
print('Fitted, a = ' + str(value(m.a)))
print('Fitted, b = ' + str(value(m.b)))
print('Fitted, c = ' + str(value(m.c)))

N = list(m.N)
x_meas = [value(m.x_meas[i]) for i in N]
y_meas = [value(m.y_meas[i]) for i in N]
y_est  = [value(m.y_est[i])  for i in N]

# plot data
plt.figure()
plt.plot(x_meas,y_meas,'ro',label='Stock Data')
plt.plot(x_meas,y_est,'bx',label='Predicted')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()
plt.show()
