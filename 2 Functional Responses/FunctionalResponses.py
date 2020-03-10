import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int

#Default params
#RM
r_d=1
m_d=0.1

#Holling params
aH_d = 3.05
bH_d = 2.68
#Ivlev params
aI_d = 1
bI_d = 2
#trig params
aT_d = 0.99
bT_d = 1.48

#bounds for x
lbound = 0.001 # not zero to avoid division by 0
ubound = 5

def defaultParams(s):
    if(s=="h"):
        return np.array([aH_d,bH_d])
    if(s=="i"):
        return np.array([aI_d,bI_d])
    if(s=="t"):
        return np.array([aT_d,bT_d])

#return the color of the model specified with "s".
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

def f(x,s,a,b):
    if(s=="h"):
        return fh(x,a,b)
    if(s=="i"):
        return fi(x,a,b)
    if(s=="t"):
        return ft(x,a,b)


def plotCompFunctions(s1=0, s2=0, s3=0, p1=0, p2=0, p3=0, bounds =[0.00001,5],
                      labels=["Holling","Ivlev","Trigonometric"],title="Functional responses"):
    x = np.linspace(bounds[0],bounds[1],bounds[1]*10000)
    if(p1 == 0):
        p1 = defaultParams(s1)
    if(p2 == 0):
        p2 = defaultParams(s2)
    if(p3 == 0):
        p3 = defaultParams(s3)

    if(s1 != 0):
        plt.plot(x,f(x, s1, p1[0], p1[1]),color(s1))
    if(s2 != 0):
        plt.plot(x,f(x, s2, p2[0], p2[1]),color(s2))
    if(s3 != 0):
        plt.plot(x,f(x, s3, p3[0], p3[1]),color(s3))
    plt.legend(labels,loc="best")
    plt.title(title)
    plt.xlabel("prey concentration, x")
    plt.ylabel("f(x)")
    plt.show()

plotCompFunctions("h","i","t")