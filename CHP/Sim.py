# Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button
'''
exp0 = np.genfromtxt('C:/data/python/chp/25T2.txt')
exp1 = np.genfromtxt('C:/data/python/chp/30T2.txt')
exp2 = np.genfromtxt('C:/data/python/chp/35T2.txt')
exp3 = np.genfromtxt('C:/data/python/chp/40T2.txt')
'''
exp0 = np.genfromtxt('25T2.txt')
exp1 = np.genfromtxt('30T2.txt')
exp2 = np.genfromtxt('35T2.txt')
exp3 = np.genfromtxt('40T2.txt')
exp0[:,1] = exp0[:,1] + 273.15
exp1[:,1] = exp1[:,1] + 273.15
exp2[:,1] = exp2[:,1] + 273.15
exp3[:,1] = exp3[:,1] + 273.15
conv = [0.713, 0.816, 0.917, 0.980]

# Parameters
Ax 	 = 5.065;   # cm2
rho  = 0.00052; # kg/cm3
Fa0  = 4.477;   # mol/hr
Ca0  = 1.923;   # mol/kg
k0   = 6200000;	#2229;    # 1/hr
Ea   = 36000;	#17857;   # J/mol
dH   = -110000;	#-361252; # J/mol
UA 	 = 45;		#69;      # J/cm2/hr/k (=191.67 W/m2/k)
Ta0  = 293.15;
mc 	 = 500;     # mol/hr
CpCP = 68.1479; # cal/mol/K
CpCM = 46.996;
CpH2 = 7.5318;
CpCA = 57.8437;
CpWR = 17.7058;
'''
# Using Estimated Parameter
paramfile = np.genfromtxt('Result_20200320-1125.out')
k0   = paramfile[0]
Ea   = paramfile[1]
dH   = paramfile[2]
UA   = paramfile[3]
'''
def ODEfun(Yfuncvec, L, Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA):
    Ta= Yfuncvec[0]
    T = Yfuncvec[1]
    X = Yfuncvec[2]
    sumcp = 4.184*(CpCP*(1 - X) + CpCP*(1.2 - X) + CpCM*3 + CpCA*X + CpWR*X);  # Initial CHP:H2:Cumene = 1:1.2:3
    dCp = 4.184*(CpCA + CpWR - CpCP - CpH2);
	# Explicit Equation Inline
    k = k0 * np.exp(-Ea/(8.314*T)); 
    ra = k * Ca0 * (1 - X);
    # Differential equations
    dTadL = 0*UA * (T - Ta) / (mc*CpWR);
    dTdL = Ax *( (UA * (Ta - T)) - rho * ra * (dH + dCp*(T-298.15)) ) / (sumcp * Fa0); 
    dXdL = Ax * rho * (ra / Fa0);
    return np.array([dTadL,dTdL,dXdL])

Lspan = np.linspace(0, 301, 1000) # Range for the independent variable
Lspan = np.hstack((Lspan,exp0[:,0],exp1[:,0],exp2[:,0],exp3[:,0]))
Lspan = np.unique(Lspan) 
Lspan = np.sort(Lspan)
y0 = np.array([298.15,298.15,0]) # Initial values for the dependent variables
y1 = np.array([303.15,303.15,0]) # Initial values for the dependent variables
y2 = np.array([308.15,308.15,0]) # Initial values for the dependent variables
y3 = np.array([313.15,313.15,0]) # Initial values for the dependent variables

sol0 = odeint(ODEfun, y0, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
sol1 = odeint(ODEfun, y1, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
sol2 = odeint(ODEfun, y2, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
sol3 = odeint(ODEfun, y3, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))

Ta =np.vstack((sol0[:, 0],sol1[:, 0],sol2[:, 0],sol3[:, 0]))
T =np.vstack((sol0[:, 1],sol1[:, 1],sol2[:, 1],sol3[:, 1]))
X =np.vstack((sol0[:, 2],sol1[:, 2],sol2[:, 2],sol3[:, 2]))

# Calculation of Fitting Error 
est0 = np.zeros(len(exp0[:,0]))
est1 = np.zeros(len(exp1[:,0]))
est2 = np.zeros(len(exp2[:,0]))
est3 = np.zeros(len(exp3[:,0]))
for i in range(0,len(exp0[:,0])):
	est0[i] = T[0,np.where(Lspan==exp0[i,0])]
for i in range(0,len(exp1[:,0])):
	est1[i] = T[1,np.where(Lspan==exp1[i,0])]
for i in range(0,len(exp2[:,0])):
	est2[i] = T[2,np.where(Lspan==exp2[i,0])]
for i in range(0,len(exp3[:,0])):
	est3[i] = T[3,np.where(Lspan==exp3[i,0])]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
ErrT = [rmse(est0,exp0[:,1]), rmse(est1,exp1[:,1]), rmse(est2,exp2[:,1]), rmse(est3,exp3[:,1])]
ErrX = np.sqrt((np.subtract([X[0,len(Lspan)-1],X[1,len(Lspan)-1],X[2,len(Lspan)-1],X[3,len(Lspan)-1]], conv)**2))

#%%
fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
plt.subplots_adjust(left  = 0.25)
fig.subplots_adjust(wspace=0.25,hspace=0.3)
fig.suptitle("""CHP Dehydration (Constant $T_a$)""", fontweight='bold', x = 0.34,y=0.97)

# 25C
ax1.plot(exp0[:,0],exp0[:,1], color='blue', linestyle='none', marker='.')
p1,p2= ax1.plot(Lspan,T[0,:],Lspan,Ta[0,:])
ax1.legend(['$T_{exp}$','T','$T_a$'], loc='upper right')
ax1.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax1.set_title('Temperature (K)', fontsize='medium')
ax1.set_ylim(298,348)
ax1.set_xlim(0,300)
ax1.grid()
ax1.ticklabel_format(style='plain',axis='x')

p3 = ax2.plot(Lspan,X[0,:])[0]
ax2.plot(300, conv[0], color='blue', linestyle='none', marker='o')
ax2.legend(['$X_{exp}$','X'], loc='lower right')
ax2.set_ylim(0,1)
ax2.set_yticks(np.arange(0,1,0.1))
ax2.set_xlim(0,300)
ax2.grid()
ax2.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax2.set_title('Conversion', fontsize='medium')
ax2.ticklabel_format(style='plain',axis='x')
ax2.text(-80, 1.03, r'T$_a$ = 25$^\circ$C', style='italic', weight='bold', color='red', fontsize=10)

# 30C
ax3.plot(exp1[:,0],exp1[:,1], color='blue', linestyle='none', marker='.')
p4,p5= ax3.plot(Lspan,T[1,:],Lspan,Ta[1,:])
ax3.legend(['$T_{exp}$','T','$T_a$'], loc='upper right')
ax3.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax3.set_title('Temperature (K)', fontsize='medium')
ax3.set_ylim(303,353)
ax3.set_xlim(0,300)
ax3.grid()
ax3.ticklabel_format(style='plain',axis='x')

p6 = ax4.plot(Lspan,X[1,:])[0]
ax4.plot(300, conv[1], color='blue', linestyle='none', marker='o')
ax4.legend(['$X_{exp}$','X'], loc='lower right')
ax4.set_ylim(0,1)
ax4.set_yticks(np.arange(0,1,0.1))
ax4.set_xlim(0,300)
ax4.grid()
ax4.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax4.set_title('Conversion', fontsize='medium')
ax4.ticklabel_format(style='plain',axis='x')
ax4.text(-80, 1.03, r'T$_a$ = 30$^\circ$C', style='italic', weight='bold', color='red', fontsize=10)

# 35C
ax5.plot(exp2[:,0],exp2[:,1], color='blue', linestyle='none', marker='.')
p7,p8= ax5.plot(Lspan,T[2,:],Lspan,Ta[2,:])
ax5.legend(['$T_{exp}$','T','$T_a$'], loc='upper right')
ax5.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax5.set_title('Temperature (K)', fontsize='medium')
ax5.set_ylim(308,358)
ax5.set_xlim(0,300)
ax5.grid()
ax5.ticklabel_format(style='plain',axis='x')

p9 = ax6.plot(Lspan,X[2,:])[0]
ax6.plot(300, conv[2], color='blue', linestyle='none', marker='o')
ax6.legend(['$X_{exp}$','X'], loc='lower right')
ax6.set_ylim(0,1)
ax6.set_yticks(np.arange(0,1,0.1))
ax6.set_xlim(0,300)
ax6.grid()
ax6.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax6.set_title('Conversion', fontsize='medium')
ax6.ticklabel_format(style='plain',axis='x')
ax6.text(-80, 1.03, r'T$_a$ = 35$^\circ$C', style='italic', weight='bold', color='red', fontsize=10)

# 40C
ax7.plot(exp3[:,0],exp3[:,1], color='blue', linestyle='none', marker='.')
p10,p11= ax7.plot(Lspan,T[3,:],Lspan,Ta[3,:])
ax7.legend(['$T_{exp}$','T','$T_a$'], loc='upper right')
ax7.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax7.set_title('Temperature (K)', fontsize='medium')
ax7.set_ylim(313,363)
ax7.set_xlim(0,300)
ax7.grid()
ax7.ticklabel_format(style='plain',axis='x')

p12 = ax8.plot(Lspan,X[3,:])[0]
ax8.plot(300, conv[3], color='blue', linestyle='none', marker='o')
ax8.legend(['$X_{exp}$','X'], loc='lower right')
ax8.set_ylim(0,1)
ax8.set_yticks(np.arange(0,1,0.1))
ax8.set_xlim(0,300)
ax8.grid()
ax8.set_xlabel(r'$Length  {(cm)}$', fontsize='medium')
ax8.set_title('Conversion', fontsize='medium')
ax8.ticklabel_format(style='plain',axis='x')
ax8.text(-80, 1.03, r'T$_a$ = 40$^\circ$C', style='italic', weight='bold', color='red', fontsize=10)

axcolor = 'black'
ax_k0 = plt.axes([0.07, 0.78, 0.1, 0.015], facecolor=axcolor)
ax_Ea = plt.axes([0.07, 0.74, 0.1, 0.015], facecolor=axcolor)
ax_dH = plt.axes([0.07, 0.70, 0.1, 0.015], facecolor=axcolor)
ax_UA = plt.axes([0.07, 0.66, 0.1, 0.015], facecolor=axcolor)
#ax_Ta0 = plt.axes([0.07, 0.62, 0.1, 0.015], facecolor=axcolor)

sk0 = Slider(ax_k0, r'k0 ($\frac{1}{hr}$)', 0, 2*k0, valinit=k0,valfmt='%1.0f')
sEa= Slider(ax_Ea, r'E$_a$ ($\frac{J}{mol}$)', 0, 2*Ea, valinit=Ea,valfmt='%1.0f')
sdH = Slider(ax_dH, r'$\Delta H_{Rx}$ ($\frac{J}{mol}$)', 0, -2*dH, valinit=-dH,valfmt='%1.0f')
sUA = Slider(ax_UA,r'Ua ($\frac{J}{cm^2.hr.K}$)', 0, 2*UA, valinit=UA,valfmt='%1.0f')
#sTa0 = Slider(ax_Ta0, r'T$_a$ ($K$)', 273, 323, valinit=298,valfmt='%1.0f')

def update_plot2(val):
	k0 =sk0.val
	dH =sdH.val
	Ea= sEa.val
	UA= sUA.val
	#Ta0= sTa0.val
	
	y0 = np.array([298.15,298.15,0]) # Initial values for the dependent variables
	y1 = np.array([303.15,303.15,0]) # Initial values for the dependent variables
	y2 = np.array([308.15,308.15,0]) # Initial values for the dependent variables
	y3 = np.array([313.15,313.15,0]) # Initial values for the dependent variables
	
	sol0 = odeint(ODEfun, y0, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
	sol1 = odeint(ODEfun, y1, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
	sol2 = odeint(ODEfun, y2, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))
	sol3 = odeint(ODEfun, y3, Lspan, (Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA))

	Ta =np.vstack((sol0[:, 0],sol1[:, 0],sol2[:, 0],sol3[:, 0]))
	T =np.vstack((sol0[:, 1],sol1[:, 1],sol2[:, 1],sol3[:, 1]))
	X =np.vstack((sol0[:, 2],sol1[:, 2],sol2[:, 2],sol3[:, 2]))
	
	p1.set_ydata(T[0,:])
	p2.set_ydata(Ta[0,:])
	p3.set_ydata(X[0,:])
	p4.set_ydata(T[1,:])
	p5.set_ydata(Ta[1,:])
	p6.set_ydata(X[1,:])
	p7.set_ydata(T[2,:])
	p8.set_ydata(Ta[2,:])
	p9.set_ydata(X[2,:])
	p10.set_ydata(T[3,:])
	p11.set_ydata(Ta[3,:])
	p12.set_ydata(X[3,:])
	fig.canvas.draw_idle()

sk0.on_changed(update_plot2)
sEa.on_changed(update_plot2)
sdH.on_changed(update_plot2)
sUA.on_changed(update_plot2)
#sTa0.on_changed(update_plot2)

resetax = plt.axes([0.07, 0.84, 0.09, 0.04])
button = Button(resetax, 'Reset variables', color='cornflowerblue', hovercolor='0.975')

def reset(event):
	sk0.reset()
	sEa.reset()
	sdH.reset()
	sUA.reset()
	#sTa0.reset()
button.on_clicked(reset)

fig.set_size_inches(18, 9)
plt.show()
