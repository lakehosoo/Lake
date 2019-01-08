import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt 
'''
# pip install uncertainties, if needed
try:
     import uncertainties.unumpy as unp
     import uncertainties as unc
except:
     import pip
     pip.main(['install','uncertainties'])
     import uncertainties.unumpy as unp
     import uncertainties as unc
'''
# import data
url = 'http://apmonitor.com/che263/uploads/Main/stats_data.txt'
data = pd.read_csv(url)
x = data['x'].values
y = data['y'].values
n = len(y)

def f(x, a, b):
     return a * x + b

popt, pcov = curve_fit(f, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y-f(x,*popt))**2)/((n-1.0)*np.var(y,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.figure()
plt.scatter(x, y, s=3, label='Data')

# calculate regression confidence interval
px = np.linspace(14, 24, 100)
py = f(px,*popt)
nom = unp.nominal_values(py)
std = unp.std_devs(py)

def predband(x, xd, yd, p, func, conf=0.95):
     # x = requested points
     # xd = x data
     # yd = y data
     # p = parameters
     # func = function name
     alpha = 1.0 - conf    # significance
     N = xd.size          # data sample size
     var_n = len(p)  # number of parameters
     # Quantile of Student's t distribution for p=(1-alpha/2)
     q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
     # Stdev of an individual measurement
     se = np.sqrt(1. / (N - var_n) * \
                  np.sum((yd - func(xd, *p)) ** 2))
     # Auxiliary definitions
     sx = (x - xd.mean()) ** 2
     sxd = np.sum(sx)
     # Predicted values (best-fit model)
     yp = func(x, *p)
     # Confidence interval
     dyc = q * se * np.sqrt((1.0/N) + (sx/sxd))
     # Prediction interval
     dyp = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
     # Upper/lower confidence/prediction bands
     lcb, ucb = yp - dyc, yp + dyc
     lpb, upb = yp - dyp, yp + dyp
     return lcb, ucb, lpb, upb

lcb, ucb, lpb, upb = predband(px, x, y, popt, f, conf=0.95)

# plot the regression
plt.plot(px, nom, c='black', label='y=a x + b')

# uncertainty lines (95% confidence)
plt.plot(px, lcb, c='orange',label='95% Confidence Region')
plt.plot(px, ucb, c='orange')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')

# save and show figure
#plt.savefig('regression.png')
plt.show()
