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
ubound = 20

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

#returns the name of the model specified with "s"
def modelName(s):
    if(s=="h"):
        return "Holling"
    if(s=="i"):
        return "Ivlev"
    if(s=="t"):
        return "trigonometric"
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
#functional response
def f(x,a,b,s):
    if(s=="h"):
        return fh(x,a,b)
    if(s=="i"):
        return fi(x,a,b)
    if(s=="t"):
        return ft(x,a,b)
    else:
        raise Exception("Bad s : choose h for Holling, i for Ivlev, t for trigonometric")
#prey growth
def g(x,params):
    return params[2]*x*(1-(x/params[3]))

#Returns nullclines of a function
def nullcl_x(x,params,s):
    return g(x,params)/f(x,params[0],params[1],s)

def nullcl_y(x,params,s):
    if(s=="h"):
        return params[4] / (params[0] - params[4] * params[1])
    elif(s=="i"):
        return -np.log(1 - (params[4] / params[0])) / params[1]
    elif(s=="t"):
        return np.arctanh(params[4]/params[0]) / (params[1])
    else:
        raise Exception("Bad s : choose h for Holling, i for Ivlev, t for trigonometric")

# return the Rosenzweig MacArthur predator (x[1]) prey (x[0]) model
def RM(t,x,p,s):
    return np.array([g(x[0], p) - f(x[0],p[0],p[1],s) * x[1],
                     f(x[0],p[0],p[1],s) * x[1] - p[4] * x[1]])

#returns the jocobi matrix
def jacobian(x,y,params,s):
    h = 0.1* (x[2]-x[1]) # in theory h is an infinitesimal.
    t = None
    d1dx = (RM(t, [x + h, y], params, s)[0] - RM(t, [x, y], params, s)[0]) / h
    d1dy = (RM(t, [x, y + h], params, s)[0] - RM(t, [x, y], params, s)[0]) / h
    d2dx = (RM(t, [x + h, y], params, s)[1] - RM(t, [x, y], params, s)[1]) / h
    d2dy = (RM(t, [x, y + h], params, s)[1] - RM(t, [x, y], params, s)[1]) / h

    return np.array([d1dx,d1dy,d2dx,d2dy])

#draws the equilibrium of the system with the functional response indicated by s
def equilibrium(x,params,s,draw=True,returnIndex=False):
    err=0.5* (x[1]-x[0])#maximum approximation error of the intersection
    circleSize = 150
    for i in range(1, len(x)):
        if(x[i] < nullcl_y(x,params,s)+err and x[i] > nullcl_y(x,params,s)-err):
            if(returnIndex):
                return i
            if(draw):
                plt.scatter(x[i], nullcl_x(x, params, s)[i], facecolors="white", edgecolors=color(s),
                            s = circleSize, zorder = 3)
            if(isStable(x,nullcl_x(x,params,s),i,params,s)):
                if(draw):
                    plt.scatter(x[i], nullcl_x(x, params, s)[i], color=color(s), s=circleSize, zorder=4)

#returns whether or not the given point of a function is stable based on the Routh-Hurwitz-criterion
def isStable(x,y,i,params,s):
    a11,a12,a21,a22 = jacobian(x,y,params,s)
    tr = a11[i] + a22[i]                    #trace of the jacobian matrix
    det = a11[i] * a22[i] - a12[i] * a21[i] #determinant

    if(tr<0 and det>0):
        return True
    else:
        return False

#finds the first K for which the system is unstable and print it.
def enrichmentResponse(x,s,Kmin=0,Kmax=13,params = 0):
    if(params == 0):
        params = defaultParams(s)
    eq_i = equilibrium(x,params,s,draw=False,returnIndex=True)
    K_tmp = 0
    prove = np.linspace(0.000001,1.5,80)
    steps = (Kmax-Kmin) * 2000  # How many K's are being evaluated

    K_linspace = np.linspace(Kmin,Kmax,steps)
    a_,b_,r_,K_,m_ = params

    #for every K check if the equilibrium is unstable.
    #Also tries to if the found K is valid by checking consistency with higher values of K
    for j in range(1, len(K_linspace)):
        params_tmp = [a_,b_,r_,K_linspace[j],m_]
        print(K_linspace[j])
        if (not isStable(x, nullcl_x(x,params_tmp,s), eq_i, params_tmp,s)):
            print("unstable")
            proved = True
            for p in prove:
                params_prove = [a_,b_,r_,K_linspace[j]+p,m_]
                if (isStable(x, nullcl_x(x, params_prove, s), eq_i, params_prove, s)):
                    proved = False
                    break
            if(proved):
                K_tmp = K_linspace[j]
                break


    print("The " + modelName(s) +" system starts being unstable for K = " + str(K_tmp))

#plots the nullclines of the functions and their corresponding equilibria
def plot_nullclines(x, y, params = None, which = ["h","i","t"], fit = False, err = 0,
                    legend = 0, title = None ,legLoc ="best"):
    if(params == None and not fit):
        params = np.array((defaultParams("h"), defaultParams("i"), defaultParams("t")))
    if(legend == 0 and not fit):
        legend = "Holling","Ivlev","Trigonometric"
    if(title == None and not fit):
        title = "Nullclines of the different systems for K = " + str(K)
    if(fit):
        fparams = params[0]
        dparams = params[1]
        title = "Nullclines: " + which[0] + '-Model fitted to ' + which[1] + "-Model K = " + str(K)\
                + "\n Data-Param: " + str(dparams) + "\n Fitted-Param: " + str(fparams) + " Error: " + str(err)
    ax = plt.subplot()
    ax.set_xscale('log')
    for axis in [ax.xaxis]:
        axis.set_major_formatter(plt.ScalarFormatter())
    i = 0
    for s in which:
        plt.plot(x, nullcl_x(x, params[i], s), color=color(s), zorder=1)
        equilibrium(x, params[i], s)
        i=i+1
    i=0
    for s in which:
        plt.axvline(nullcl_y(x, params[i], s), color=color(s), zorder=2)
        i = i + 1

    if(legend != None):
        plt.legend(legend, loc=legLoc)
    plt.xticks([0.01,0.05,0.25])
    plt.xticks([0.01,0.1,1,5,20],[0.01,0.1,1,5,20])
    plt.yticks([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2],[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2])
    plt.ylim(0, 1.1)
    plt.xlim(0.01,5)
    plt.title(title,size =13)
    plt.ylabel("predator concentration, y ", fontsize = 12)
    plt.xlabel("prey concentration, x", fontsize = 12)
    plt.show()

#Main
K=4
x=np.linspace(lbound,ubound,200000)
y=np.linspace(lbound,ubound,200000)
hollingParamters = [1.47335,0.12232,1,4,0.1]
ivlevParamters = [6.07528,0.24252,1,4,0.1]
trigParamters = defaultParams("t")
p = [trigParamters,ivlevParamters]
#enrichmentResponse(x,"h",Kmin=0.4)
#enrichmentResponse(x,"i",Kmin=1)
#enrichmentResponse(x,"t",Kmin = 10.06)
plot_nullclines(x,y,which=["t","i"],legend = ("Trigonometric ","Ivlev (fitted)"),params=p,title="")

