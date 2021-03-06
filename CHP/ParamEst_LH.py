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

# Data Treatment - Temperature
exp0 = np.delete(exp0, np.argwhere((exp0 >= 150) & (exp0 <= 301)), axis=0)
exp1 = np.delete(exp1, np.argwhere((exp1 >= 150) & (exp1 <= 301)), axis=0)
exp2 = np.delete(exp2, np.argwhere((exp2 >= 150) & (exp2 <= 301)), axis=0)
exp3 = np.delete(exp3, np.argwhere((exp3 >= 150) & (exp3 <= 301)), axis=0)

# Parameters
Ax 	 = 5.065    # cm2
rho  = 0.00052  # kg/cm3
Fa0  = 4.477    # mol/hr
Ca0  = 1.923    # mol/kg
k0   = 2229     # 1/hr
Ea   = 17857    # J/mol
dH   = -361252  # J/mol
Ta0  = 293.15
UA 	 = 69       # J/cm2/hr/k (=191.67 W/m2/k)
mc 	 = 500      # mol/hr
CpCP = 68.1479  # cal/mol/K
CpCM = 46.996
CpH2 = 7.5318
CpCA = 57.8437
CpWR = 17.7058
Kc   = 1000

Lspan = np.linspace(0, 301, 1000) # Range for the independent variable
Lspan = np.hstack((Lspan,exp0[:,0],exp1[:,0],exp2[:,0],exp3[:,0]))
Lspan = np.unique(Lspan) 
Lspan = np.sort(Lspan)
y0 = np.array([298.15,298.15,0]) # Initial values for the dependent variables
y1 = np.array([303.15,303.15,0]) # Initial values for the dependent variables
y2 = np.array([308.15,308.15,0]) # Initial values for the dependent variables
y3 = np.array([313.15,313.15,0]) # Initial values for the dependent variables

def ODEfun(Yfuncvec, L, Ax,Fa0,rho,Ca0,k0,Ea,dH,Ta0,UA,mc,CpCP,CpCM,CpH2,CpCA,Kc):
    Ta= Yfuncvec[0]
    T = Yfuncvec[1]
    X = Yfuncvec[2]
    sumcp = 4.184*(CpCP*(1 - X) + CpCP*(1.2 - X) + CpCM*3 + CpCA*X + CpWR*X)  # Initial CHP:H2:Cumene = 1:1.2:3
    dCp = 4.184*(CpCA + CpWR - CpCP - CpH2)
    # Explicit Equation Inline
    k = k0 * np.exp(-Ea/(8.314*T))
    #ra = k * Ca0 * (1 - X)
    ra = k * Ca0 * (1 - ((1 + 1/Kc)*X))
    # Differential equations
    dTadL = 0*UA * (T - Ta) / (mc*CpWR)
    dTdL = Ax *( (UA * (Ta - T)) - rho * ra * (dH + dCp*(T-298.15)) ) / (sumcp * Fa0)
    dXdL = Ax * rho * (ra / Fa0)
    return np.array([dTadL,dTdL,dXdL])

def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def error(param):
    sol0 = odeint(ODEfun, y0, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
    sol1 = odeint(ODEfun, y1, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
    sol2 = odeint(ODEfun, y2, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
    sol3 = odeint(ODEfun, y3, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
    #Ta =np.vstack((sol0[:, 0],sol1[:, 0],sol2[:, 0],sol3[:, 0]))
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
    
    ErrT = [mse(est0,exp0[:,1]), mse(est1,exp1[:,1]), mse(est2,exp2[:,1]), mse(est3,exp3[:,1])]
    ErrX = 10000*(np.subtract([X[0,len(Lspan)-1],X[1,len(Lspan)-1],X[2,len(Lspan)-1],X[3,len(Lspan)-1]], conv)**2)
    #print(ErrX,ErrT)
    
    return -(np.sum(ErrX**2)*err_w_conv + np.sum(ErrT))

def cal_pop_fitness(pop):
    # Calculating the fitness value of each solution in the current population.
    sumoferr = np.zeros(sol_per_pop)
    for i in range(0,sol_per_pop):
        param = pop[i,:]
        sumoferr[i] = error(param)
    sumoferr[np.isnan(sumoferr)] = -1000000
    return sumoferr

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -1000000
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        # The point at which crossover takes place between two parents.
        crossover_point = np.random.randint(parents.shape[1])
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations):
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        for mutation_num in range(num_mutations):
            gene_idx = np.random.randint(offspring_crossover.shape[1])
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx]*(1 + ratio_mutation*random_value)
    return offspring_crossover

# Parameters to Optimize : k0, Ea, dH, UA 
#initials = [2229, 17857, -361252, 69]
#initial_best = [6200000, 36000, -110000, 45]
initial_max = [6000000, 50000, -500000, 100, 100000]

# Number of Parameters to optimize.
num_params = len(initial_max)

# Genetic algorithm parameters:
sol_per_pop = 100		# Population size
num_parents_mating = 30	# Mating pool size
num_mutations = 3		# Number of times being mutated
ratio_mutation = 0.5    # Degree of mutation : 1 for -100% ~ 100%
num_generations = 3000  # Total number of generations
err_w_conv = 1          # Weight for conversion to calculate fitting error

# Defining the population size.
pop_size = (sol_per_pop,num_params) # The population will have sol_per_pop chromosome where each chromosome has num_params genes.
# Creating the initial population.
initial_factor = np.random.uniform(low=0, high=1, size=pop_size)
new_population = initial_factor*initial_max
#new_population[0,:] = initial_best
print(new_population)

best_outputs = []
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_population)
    print("Fitness")
    print(fitness)

    best_outputs.append(np.max(fitness))
    # The best result in the current iteration.
    print("Best result : ", np.max(fitness))
    
    # Convergence Test
    if (generation%(num_generations//20) == 0):
        if (generation//(num_generations//10) > 3):
            if (best_outputs[generation] - best_outputs[generation - num_generations//10]) < 0.001:
                ratio_mutation = ratio_mutation/2
        #if (generation//(num_generations//10) > 6):
        #    if (best_outputs[generation] - best_outputs[generation - num_generations//10]) < 0.000001:
        #        break

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, 
                                      num_parents_mating)
    #print("Parents")
    #print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_params))
    #print("Crossover")
    #print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations)
    #print("Mutation")
    #print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Final Result
fitness = cal_pop_fitness(new_population)
best_match_idx = np.where(fitness == np.max(fitness))
print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

# Current Time 
from datetime import datetime
now = datetime.now()
now = now.strftime('%Y%m%d-%H%M')

# Plot #1
fig, ax = plt.subplots()
ax.plot(best_outputs)
ax.set_xlabel("Iteration")
ax.set_ylabel("Fitness")
ax.text(10, -4,
		 r'Best Fitness = 'f'{fitness[best_match_idx]}'
         '\n'
         r'Generation # = 'f'{generation}'
         '\n'
         r'R_Mutation = 'f'{ratio_mutation}'
		 , ha='left', wrap = True, fontsize=10,fontweight='normal')
fig.savefig('Result_'+now+'.png', dpi=300)
#plt.show()

# Variables for Plot #2
param = np.ravel(new_population[best_match_idx,:])
sol0 = odeint(ODEfun, y0, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
sol1 = odeint(ODEfun, y1, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
sol2 = odeint(ODEfun, y2, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))
sol3 = odeint(ODEfun, y3, Lspan, (Ax,Fa0,rho,Ca0,param[0],param[1],param[2],Ta0,param[3],mc,CpCP,CpCM,CpH2,CpCA,param[4]))

Ta =np.vstack((sol0[:, 0],sol1[:, 0],sol2[:, 0],sol3[:, 0]))
T =np.vstack((sol0[:, 1],sol1[:, 1],sol2[:, 1],sol3[:, 1]))
X =np.vstack((sol0[:, 2],sol1[:, 2],sol2[:, 2],sol3[:, 2]))

# Calculation of Fitting Error #########################################################
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
ErrT = [mse(est0,exp0[:,1]), mse(est1,exp1[:,1]), mse(est2,exp2[:,1]), mse(est3,exp3[:,1])]
ErrX = 10000*(np.subtract([X[0,len(Lspan)-1],X[1,len(Lspan)-1],X[2,len(Lspan)-1],X[3,len(Lspan)-1]], conv)**2)
###########################################################################################

# Plot #2
fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
plt.subplots_adjust(left  = 0.25)
fig.subplots_adjust(wspace=0.25,hspace=0.3)
fig.suptitle('CHP Dehydration (Constant $T_a$, Total Err = ErrT + 'f'{err_w_conv:.0f}'' x ErrX)', fontweight='bold', x = 0.34,y=0.97)

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
ax1.text(10, 340, r'MSE = 'f'{ErrT[0]:.1f}', color='blue', fontsize=10)

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
ax2.text(10, 0.85, r'MSE = 'f'{ErrX[0]:.1f}', color='blue', fontsize=10)

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
ax3.text(10, 345, r'MSE = 'f'{ErrT[1]:.1f}', color='blue', fontsize=10)

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
ax4.text(10, 0.85, r'MSE = 'f'{ErrX[1]:.1f}', color='blue', fontsize=10)

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
ax5.text(10, 350, r'MSE = 'f'{ErrT[2]:.1f}', color='blue', fontsize=10)

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
ax6.text(10, 0.85, r'MSE = 'f'{ErrX[2]:.1f}', color='blue', fontsize=10)

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
ax7.text(10, 355, r'MSE = 'f'{ErrT[3]:.1f}', color='blue', fontsize=10)

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
ax8.text(10, 0.85, r'MSE = 'f'{ErrX[3]:.1f}', color='blue', fontsize=10)

ax1.text(-400, 300,
		 r'k0  = 'f'{param[0]:.0f} (1/hr)'
		 '\n'
		 r'E$_{a}$  = 'f'{param[1]:.0f} (J/mol)'
		 '\n'
		 r'dH = 'f'{param[2]:.0f} (J/mol)'
		 '\n'
		 r'UA = 'f'{param[3]:.0f} (J/cm2/hr/K)'
		 , ha='left', wrap = True, fontsize=12,
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0), fontweight='normal')

np.savetxt('Result_'+now+'.out', param, delimiter=',')

#plt.show()
fig.set_size_inches(18, 9)
fig.savefig('Result_'+now+'_Pic.png', dpi=300)
