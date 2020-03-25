from numpy.random import *
import numpy as np

NO_PARAM_DISTRI =[
    rand,
    chisquare,
    exponential,
    #standard_cauchy,
    standard_exponential,
]

ONE_PARAM_DISTRI = [
    #pareto,
    poisson,
    power,
    rayleigh,
    #standard_gamma,
    #standard_t,
    chisquare,
    #weibull
    lambda p, size: (rand(size) > p)
]

TWO_PARAM_DISTRI = [
    #beta,
    #f,
    #gamma,
    #gumbel,
    laplace,
    logistic,
    #lognormal,
    normal,
    #vonmises,
    #wald    
]

PROB_ONE_PARAM_DSITRI = [
    #geometric,
    #logseries
]

TINY = 0.0001

# size: int
def gen(size):
    i = randint(3)
    if i == 0:
        j = randint(len(NO_PARAM_DISTRI))
        res = NO_PARAM_DISTRI[j](size)
    elif i == 1:
        j = randint(len(ONE_PARAM_DISTRI))
        a = 2*rand()+TINY
        res = ONE_PARAM_DISTRI[j](a, size)
    elif i == 2:
        j = randint(len(TWO_PARAM_DISTRI))
        a = 5*rand()+TINY
        b = rand()+TINY
        res = TWO_PARAM_DISTRI[j](a,b, size)
    else:
        j = randint(len(PROB_ONE_PARAM_DSITRI))
        p = rand()
        res = PROB_ONE_PARAM_DSITRI[j](p, size)
    
    res = np.maximum(res, 0)
    res = res/np.max(res)
    
    return res


def V_GEN(x,y, Vmax=1):
    res = np.empty([x,y])
    for i in range(x):
        res[i] = Vmax*gen(y)
      
    return res