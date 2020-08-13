# Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#pd.read_excel('HME/HME_Data_V2.xlsx', sheet_name='Rearrange', index_col=0, header=None, skiprows=2)
exp0 = np.genfromtxt('Study2_Data.txt')
exp0[:,1] = exp0[:,1] + 273.15
conv = [0.713, 0.816, 0.917, 0.980]

# Parameters
Ax   = 5.065    # cm2
rho  = 0.00052  # kg/cm3
Fa0  = 4.477    # mol/hr
Ca0  = 1.923    # mol/kg
k0   = 6200000  #2229     # 1/hr
Ea   = 36000    #17857    # J/mol
dH   = -361252  #-361252  # J/mol
UA   = 45       #69;      # J/cm2/hr/k (=191.67 W/m2/k)
Ta0  = 293.15
mc   = 500      # mol/hr
CpCP = 68.1479  # cal/mol/K
CpCM = 46.996
CpH2 = 7.5318
CpCA = 57.8437
CpWR = 17.7058

# Using Estimated Parameter
paramfile = np.genfromtxt('Study2_Param.txt')
k0   = paramfile[0]
Ea   = paramfile[1]
#dH   = paramfile[2]
UA   = paramfile[2]

def ODEfun(Yfuncvec, L, Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA):
    Ta= Yfuncvec[0]
    T = Yfuncvec[1]
    X = Yfuncvec[2]
    sumcp = 4.184*(CpCP*(1 - X) + CpCP*(1.2 - X) + CpCM*3 + CpCA*X + CpWR*X)  # Initial CHP:H2:Cumene = 1:1.2:3
    dCp = 4.184*(CpCA + CpWR - CpCP - CpH2)
    # Explicit Equation Inline
    k = k0 * np.exp(-Ea/(8.314*T)) 
    ra = k * Ca0 * (1 - X)
    # Differential equations
    dTadL = 0*UA * (T - Ta) / (mc*CpWR)
    dTdL = Ax *( (UA * (Ta - T)) - rho * ra * (dH + dCp*(T-298.15)) ) / (sumcp * Fa0) 
    dXdL = Ax * rho * (ra / Fa0)
    return np.array([dTadL,dTdL,dXdL])

Lspan = np.linspace(0, 301, 1000) # Range for the independent variable

y0 = np.array([298.15,298.15,0]) # Initial values for the dependent variables

sol0 = odeint(ODEfun, y0, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))


fig, ax = plt.subplots()

ax.plot(exp0[:,0],exp0[:,1], color='blue', linestyle='none', marker='.')
p1,p2= ax.plot(Lspan,T[0,:],Lspan,Ta[0,:])
ax.legend(['$T_{exp}$','T','$T_a$'], loc='upper right')
ax.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax.set_title('Temperature (K)', fontsize='medium')
ax.set_ylim(298,348)
ax.set_xlim(0,300)
ax.grid()
ax.ticklabel_format(style='plain',axis='x')
