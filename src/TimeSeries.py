import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int

np.set_printoptions(threshold=np.inf)

#params
#RM
r=1
m=0.1


#Holling params
aH = 3.05
bH = 2.68
#Ivlev params
aI = 1
bI = 2
#trig params
aT = 0.99
bT = 1.48

def defaultParameters(s):
    if(s=="h"):
        return [aH, bH, r, K, m]
    if(s=="i"):
        return [aI, bI, r, K, m]
    if(s=="t"):
        return [aT, bT, r, K, m]
def color(s):
    if(s=="h"):
        return "blue"
    if(s=="i"):
        return "black"
    if(s=="t"):
        return "red"

#Component functions
#Holling
def fh(x,a,b):
    return a*x/(1+b*x)
#Ivlev
def fi(x,a,b):
    return a*(1 - np.exp(-b*x))
#Trigonometric
def ft(x,a,b):
    return a*np.tanh(b*x)

def f(x,a,b,s):
    if(s=="h"):
        return fh(x,a,b)
    if(s=="i"):
        return fi(x,a,b)
    if(s=="t"):
        return ft(x,a,b)

def g(x,K_,r_):
    return r_*x*(1-(x/K_))

def RM(t,x,p,s):
    return np.array([g(x[0], p[3], p[2]) - f(x[0],p[0],p[1],s) * x[1],
                     f(x[0],p[0],p[1],s) * x[1] - p[4] * x[1]])

#Plots the time series of the populations
#population[0] = plot prey population
#population[1] = plot predator population
#which specifies which functional response was used.
#system paramters can be given by p
#param sd sets the standart deviation of the added noise. default is 0
def plotTimeSeries(population=[True,True],which=["h","i","t"],p=0,
                   legend=("Holling", "Ivlev", "Trigonometric"), legLoc="best",
                   title1="Prey density over time", title2= "Predator density over time",
                   ylim1= None, ylim2=None, xlim=(970,4010), sd=0):
    if (p == 0):
        p = []
        for i in range(0,len(which)):
            p.append(defaultParameters(which[i]))
    if(len(p) != len(which)):
        raise Exception("Dimension mismatch")

    i=0
    solved = np.empty(0)
    for s in which:
        ode = int.solve_ivp(RM,bounds,x0,max_step=0.1,args=(p[i],s),t_eval=t_eval)
        solved = np.append(solved,ode)
        i += 1

    if(sd!=0):
        noise = [np.random.normal(scale=sd, size=len(solved[0].y[0])),
                 np.random.normal(scale=sd, size=len(solved[0].y[1]))]

        for ode in solved:
            ode.y = ode.y + noise

    if(population[0]):
        i=0
        for ode in solved:
            plt.plot(ode.t, ode.y[0], color=color(which[i]))
            i +=1
        if(legend != None):
            plt.legend(legend, loc=legLoc)
        plt.title(title1)
        plt.xlabel("time, t", fontsize = 12)
        plt.ylabel("prey conentration", fontsize = 12)
        #plt.yticks([0, 0.05,0.1,0.15,0.2])
        plt.ylim(ylim1)
        plt.xlim(xlim)
        plt.show()

    if(population[1]):
        i=0
        for ode in solved:
            plt.plot(ode.t, ode.y[1], color=color(which[i]))
            i +=1

        if (legend != None):
            plt.legend(legend, loc=legLoc)
        plt.title(title2)
        plt.xlabel("time, t", fontsize = 12)
        plt.ylabel("predator conentration", fontsize = 12)
        plt.ylim(ylim2)
        plt.xlim(xlim)
        plt.show()
      
K=1
range_=[0.0001,5]
x=np.linspace(range_[0], range_[1],50000)
y=np.linspace(range_[0], range_[1],50000)
pa = [[1.47335,0.12232, r, K, m] ,[6.07528,0.24252, r, K, m],defaultParameters("t")]

x0=[0.1,1]
bounds = [1000,4000]
t_eval = np.linspace(bounds[0],bounds[1],10000)
plotTimeSeries(legend=None,which=["h","t"],sd=0.1)