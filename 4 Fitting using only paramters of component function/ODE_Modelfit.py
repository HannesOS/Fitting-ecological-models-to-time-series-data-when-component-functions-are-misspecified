import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import scipy.optimize as opt
from lmfit import Model



baseModel = ""
dataModel = ""
data = 0
counter=0
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

def fit_solveODE(x, a_, b_, r_, K_, m_):
    global baseModel
    p = [a_,b_,r_,K_,m_]
    print(np.round(p,5))
    ode = int.solve_ivp(RM,bounds,x0,max_step=0.5,args=(p,baseModel),t_eval=t_eval_).y[1]
    return ode


def data_solveODE(x,p):
    global dataModel
    return int.solve_ivp(RM,bounds,x0,max_step=0.1,args=(p,dataModel),t_eval=t_eval_)

def modelFit(base_s,data_s):
    global baseModel, dataModel, data
    baseModel = base_s
    dataModel = data_s
    data = data_solveODE([x,y],defaultParameters(data_s))
    paramBounds = (0.0001,[12,12,5,8,3])
    fit = opt.curve_fit(fit_solveODE,data.t,data.y[1],
                        bounds =paramBounds, verbose = 2)
    print(fit)




def printout(params, iter, resid, *args, **kws):
    par=np.array(list(params.valuesdict().values()))
    print(np.round(par,4))


def startingparameters(s,onlyAandB):
    if(s=="h"):
        if(onlyAandB):
            return [aH,bH]
        else:
            return [aH,bH,r,K,m]

    if(s=="i"):
        if (onlyAandB):
            return [aI, bI]
        else:
            return [aI, bI, r, K, m]
    if(s=="t"):
        if (onlyAandB):
            return [aT, bT]
        else:
            return [aT, bT, r, K, m]

def plotTimeSeries(which=[True,True],p=0,showLegend=True):
    if(which[0]):
        if(p==0):
            p = np.array([defaultParameters("h"),defaultParameters("i"),defaultParameters("t")])
        xh = int.solve_ivp(RM,bounds,x0,max_step=0.5,args=(p[0],"h"))
        xi = int.solve_ivp(RM,bounds,x0,max_step=0.5,args=(p[1],"i"))
        xt = int.solve_ivp(RM,bounds,x0,max_step=0.5,args=(p[2],"t"))
        plt.plot(xh.t,xh.y[0], color = color("h"))
        plt.plot(xi.t, xi.y[0], color = color("i"))
        plt.plot(xt.t, xt.y[0], color = color("t"))
        if (showLegend):
            plt.legend(("Holling", "Ivlev", "Trigonometric"), loc="upper right")
        plt.title("Prey density over time \n K = " +
                  str(K) + ". Initial conditions : x= " + str(x0[0]) + ", y = " + str(x0[1]))
        plt.xlabel("time, t")
        plt.ylabel("prey conentration")
        plt.show()

    if(which[1]):
        plt.plot(xh.t, xh.y[1], color = color("h"))
        plt.plot(xi.t, xi.y[1], color = color("i"))
        plt.plot(xt.t, xt.y[1], color = color("t"))
        if(showLegend):
            plt.legend(("Holling", "Ivlev", "Trigonometric"), loc="upper right")
        plt.title("Predator density over time \n K = " +
                  str(K) + ". Initial conditions: x = " + str(x0[0]) + ", y = " + str(x0[1]))
        plt.xlabel("time, t")
        plt.ylabel("predator conentration")
        plt.show()


K=4
range=[0.0001,5]
x=np.linspace(range[0], range[1],50000)
y=np.linspace(range[0], range[1],50000)
x0=[0.1,1]
bounds = [1000,4000]
t_eval_ = y=np.linspace(bounds[0], bounds[1],10000)
modelFit("h","i")


#plotTimeSeries()
