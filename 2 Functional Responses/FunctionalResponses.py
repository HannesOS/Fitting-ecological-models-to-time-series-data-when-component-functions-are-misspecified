import numpy as np
import matplotlib.pyplot as plt

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

def defaultParams(s,format = "abrkm"):
    if(format == "abrkm"):
        if (s == "h"):
            return np.array([aH_d, bH_d, r_d, K, m_d])
        if (s == "i"):
            return np.array([aI_d, bI_d, r_d, K, m_d])
        if (s == "t"):
            return np.array([aT_d, bT_d, r_d, K, m_d])
    elif(format == "ab"):
        if (s == "h"):
            return np.array([aH_d, bH_d])
        if (s == "i"):
            return np.array([aI_d, bI_d])
        if (s == "t"):
            return np.array([aT_d, bT_d])
    else:
        raise Exception("Default parameters are only defined for the formats 'abrkm' and 'ab'")

#return the color of the model specified with "s".
def color(s):
    if(s=="h"):
        return "blue"
    if(s=="i"):
        return "black"
    if(s=="t"):
        return "red"
    else:
        raise Exception("Bad s : choose h for Holling, i for Ivlev, t for trigonometric")

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
    else:
        raise Exception("Bad s : choose h for Holling, i for Ivlev, t for trigonometric")


def plotCompFunctions(models, params=None, bounds =[0.00001,5],
                      labels=["Holling","Ivlev","Trigonometric"],title="Functional responses",legLoc="best"):
    x = np.linspace(bounds[0],bounds[1],bounds[1]*10000)

    if (params == None):
        params = np.empty(0)
        for s in models:
            params = np.append(params,[defaultParams(s,"ab")])
    print(params)

    i=0
    for s in models:
        plt.plot(x,f(x,s,params[i],params[i+1]),color(s))
        i=i+2

    plt.legend(labels,loc=legLoc)
    plt.title(title,fontsize =13)
    plt.xlabel("prey concentration, x",fontsize = 12)
    plt.ylabel("f(x)",fontsize = 13)
    plt.show()

plotCompFunctions(["h","i","t"],legLoc="lower right")