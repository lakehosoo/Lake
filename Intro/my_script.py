print("Hello, world")

for i in range(20):
	if i % 3 == 0:
		print(i)
		if i % 5 == 0:
			print("Bingo!")
	print("---")

####### Basics ####### 
import numpy as np
x = np.pi

print(str(x))              # 3.141592653589793
print('{:.2f}'.format(x))  # 3.14
print('{:.10f}'.format(x)) # 3.1415926536
print('{:.5e}'.format(x))  # 3.14159e+00
print('{:.3%}'.format(x))  # 314.159%
print('{:.2f} {:.3f}'.format(x,x))  # 3.14
print('%.2f %.3f' %(x,x))  # 3.14

####### Conditionals #######
from random import *
answer = randint(0,100)
correct = False

while not(correct):
    guess = int(input("Guess a number: "))
    if guess < answer:
        print("Too low")
    elif guess > answer:
        print("Too high")
    else:
        print("Correct!")
        correct = True

####### Functions 1/2 #######
import random

y = [random.random()*100.0 for i in range(10)]
print("Print y")
print(y)

print("Sorted List")
for i in range(len(y)):
    print("%.2f" % y[i])

def avg(x):
    return sum(x) / len(x)

print("Avg: " + str(avg(y)))
print("Max: " + str(max(y)))
print("Min: " + str(min(y)))
z = sum(1 if i<50.0 else 0 for i in y)
print("Number Below 50: " + str(z))

# Another Method with NumPy
import numpy as np
print(np.mean(y))
print(np.average(y))
print(np.std(y))
print(np.median(y))

####### Functions 2/2 #######
def P_RK_IG(V, T, do_ideal_gas=False):
    R = 0.0821  # L-atm/K
    Pc = 37.2   # atm
    Tc = 132.5  # K

    a = 0.427 * pow(R,2) * pow(Tc,2.5) / Pc
    b = 0.0866 * R * Tc / Pc

    # Compute in atm
    P_ig = R * T / V
    P_rk = R * T / (V-b) - a/(V*(V+b)*pow(T,0.5))

    # Convert to Pascals
    if do_ideal_gas:
        return P_ig * 101325
    else:
        return P_rk * 101325

for T in range(490,511,10):
    V = 4.0
    while V < 8:
        print("----- Temperature: " + str(T) + " K")
        print("P_ig: " + str(P_RK_IG(V,T,True)) + " Pa")
        print("P_rk: " + str(P_RK_IG(V,T)) + " Pa")
        V = V + 2.0

####### Generating Plots #######
import numpy as np
x = np.linspace(0,6,100)
y = np.sin(x)
z = np.cos(x)

import matplotlib.pyplot as plt
plt.plot(x,y,'r--',linewidth=3)
plt.plot(x,z,'k:',linewidth=2)
plt.legend(['y','z'])
plt.xlabel('x')
plt.ylabel('values')
plt.xlim([0, 3])
plt.ylim([-1.5, 1.5])
plt.savefig('myFigure.png')
plt.savefig('myFigure.eps')
plt.show()

####### Interaction #######
myName = input('Name: ')
try:
    myAge = int(input('Age: '))
except:
    print('Invalid age, please enter a number')

####### Solve Equations #######
import numpy as np

A = np.array([ [3,-9], [2,4] ])
b = np.array([-42,2])
z = np.linalg.solve(A,b)
print(z)

M = np.array([ [1,-2,-1], [2,2,-1], [-1,-1,2] ])
c = np.array([6,1,1])
y = np.linalg.solve(M,c)
print(y)

def myFunction(z):
    x = z[0]
    y = z[1]
    w = z[2]

    F = np.empty((3))
    F[0] = pow(x,2)+pow(y,2)-20
    F[1] = y - pow(x,2)
    F[2] = w + 5 - x*y
    return F

from scipy.optimize import*

zGuess = np.array([1,1,1])
z = fsolve(myFunction,zGuess)
print(z)

####### Data Analysis #######

