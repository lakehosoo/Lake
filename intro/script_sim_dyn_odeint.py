import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def tank(h,t):
    # constants
    c1 = 0.13
    c2 = 0.20
    Ac = 2      # m^2
    # inflow
    qin = 0.5   # m^3/hr
    # outflow
    qout1 = c1 * h[0]**0.5
    qout2 = c2 * h[1]**0.5
    # differential equations
    dhdt1 = (qin   - qout1) / Ac
    dhdt2 = (qout1 - qout2) / Ac
    # overflow conditions
    if h[0]>=1 and dhdt1>=0:
        dhdt1 = 0
    if h[1]>=1 and dhdt2>=0:
        dhdt2 = 0
    dhdt = [dhdt1,dhdt2]
    return dhdt

# integrate the equations
t = np.linspace(0,10) # times to report solution
h0 = [0,0]            # initial conditions for height
y = odeint(tank,h0,t) # integrate

# plot results
plt.figure()
plt.plot(t,y[:,0],'b-')
plt.plot(t,y[:,1],'r--')
plt.xlabel('Time (hrs)')
plt.ylabel('Height (m)')
plt.legend(['h1','h2'])
plt.show()
