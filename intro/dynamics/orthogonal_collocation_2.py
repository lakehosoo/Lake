# Solve tau dx/dt = -x + k u
import numpy as np
from numpy.linalg import inv

## problem statement
x0 = 0.0  # initial condition
tf = 10.0 # final time
tau = 5.0 # model parameter (time constant)
k = 2.0   # gain

## analytic solution
time = np.linspace(0,tf)
x = k*(1.0-np.exp(-time/tau))

## numeric solutions
# 2nd order polynomial (3 points)
t2 = tf * np.array([0.0,0.5,1.0])
N2 = np.array([[0.75,-0.25], \
               [1.00, 0.00]])
M2 = inv(tf * N2)
# Solve linear system of equations by matrix manipulation
P2 = inv(tau*M2 + np.eye(2))
Q2 = np.dot(tau*M2,np.ones(2)*x0) + k*np.ones(2)
v2 = np.dot(P2, Q2)
x2 = np.insert(v2,0,x0)

# 3rd order polynomial (4 points)
t3 = tf * np.array([0.0, \
                    1.0/2.0-np.sqrt(5.0)/10.0, \
                    1.0/2.0+np.sqrt(5.0)/10.0, \
                    1.0])
N3 = np.array([[0.436,-0.281, 0.121], \
               [0.614, 0.064, 0.046], \
               [0.603, 0.230, 0.167]])
M3 = inv(tf * N3)
# Solve linear system of equations by matrix manipulation
P3 = inv(tau*M3 + np.eye(3))
Q3 = np.dot(tau*M3,np.ones(3)*x0) + k*np.ones(3)
v3 = np.dot(P3, Q3)
x3 = np.insert(v3,0,x0)

# 4th order polynomial (5 points)
t4 = tf * np.array([0.0, \
                    1.0/2.0-np.sqrt(21.0)/14.0, \
                    1.0/2.0, \
                    1.0/2.0+np.sqrt(21.0)/14.0, \
                    1.0])
N4 = np.array([[0.278, -0.202, 0.169, -0.071], \
               [0.398,  0.069, 0.064, -0.031], \
               [0.387,  0.234, 0.278, -0.071], \
               [0.389,  0.222, 0.389,  0.000]])
M4 = inv(tf * N4)
# Solve linear system of equations by matrix manipulation
P4 = inv(tau*M4 + np.eye(4))
Q4 = np.dot(tau*M4,np.ones(4)*x0) + k*np.ones(4)
v4 = np.dot(P4, Q4)
x4 = np.insert(v4,0,x0)

# compare results
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time,x,'r-',linewidth=3)
plt.plot(t2,x2,'b-.',linewidth=2,markersize=20)
plt.plot(t3,x3,'k:.',linewidth=2,markersize=20)
plt.plot(t4,x4,'g.-',linewidth=2,markersize=20)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(['Exact','3 Points','4 Points','5 Points'])
plt.text(4,0.5,'tau dx/dt = -x + k u')

plt.subplot(2,1,2)
# exact solutions at collocation points
y2 = k*(1-np.exp(-t2/tau))
y3 = k*(1-np.exp(-t3/tau))
y4 = k*(1-np.exp(-t4/tau))
plt.plot(t2,x2-y2,'b--.',linewidth=2,markersize=20)
plt.plot(t3,x3-y3,'k:.',linewidth=2,markersize=20)
plt.plot(t4,x4-y4,'g.-',linewidth=2,markersize=20)
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend(['3 Points','4 Points','5 Points'])
plt.show()
