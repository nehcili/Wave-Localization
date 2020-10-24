############################################
# Libraries
############################################

# data container
import numpy as np

# LDL decomposition
from scipy.linalg import ldl

# Solving Hu = 1
from petsc4py import PETSc

# random numbers
from random import choice

# io
from iodata import save_data_set



############################################
# constants
############################################
# this is the tiny of np.float64 as give by np.finfo(np.float32)
# note that tf uses float32 instead, but we always compute N_V(E) before storing
TINY = 1.0000000000000001e-15




############################################
# potential
############################################

# dict to store index and name of potentials
potential_dict = [
    "uniform",
    "speckle",
    "bernoulli",
    "exponential",
    "poisson",
    "power",
    "rayleigh",
    "constant",
    "chisquare",
]
bounded_potential = [
    "uniform",
    "bernoulli",
    "power",
]

# All bounded potentials have pmax = 1
# All unbounded potentials have mean = 1
# input:
#   ptype: string
#   domain_size: int
#   arg: float64
#   seed: int
# rtype:
#   np.float64
# default to uniform potential if there is no match for ptype
def gen_potential(ptype: str, domain_size: int, arg=0.0, seed=0) -> np.ndarray:
    if seed:
        np.random.seed(seed)

    if ptype == "speckle":
        # needs to get packages for the speckle potential
        return np.random.rand(domain_size)
    elif ptype == "bernoulli":
        if not arg:
            arg = 0.5 # probability of 0
        return (np.random.rand(domain_size) < arg).astype(np.float64)
    elif ptype == "chisquare":
        return np.random.chisquare(1, size=domain_size)
    elif ptype == "exponential":
        return np.random.standard_exponential(domain_size)
    elif ptype == "poisson":
        return np.random.poisson(size=domain_size)
    elif ptype == "power":
        if not arg:
            arg = 1 # x^arg in pdf
        return np.random.power(arg, size=domain_size)
    elif ptype == "rayleigh":
        sigma = np.sqrt(2/np.pi)
        return np.random.rayleigh(scale=sigma, size=domain_size)
    elif ptype == "constant":
        return np.ones(domain_size)
    elif ptype == "uniform":
        return np.random.rand(domain_size)
    else:
        return np.random.rand(domain_size)


############################################
# IDOS and Landscape u computation
############################################

# computes the landscape function u
# input:
#   np.array 1D
# rvalue:
#   np.array 1D
def compute_u(p: np.ndarray) -> np.ndarray:
    # First, make PETc Hamiltonian
    # H = -\Delta + p
    # Note: cannot use dense np.array one
    n = len(p)
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setUp()

    rstart, rend = A.getOwnershipRange()

    # first row
    if rstart == 0:
        A[0, :2] = [2+p[0], -1]
        rstart += 1
    # last row
    if rend == n:
        A[n-1, -2:] = [-1, 2+p[n-1]]
        rend -= 1
    # other rows
    for i in range(rstart, rend):
        A[i, i-1:i+2] = [-1, 2+p[i], -1]

    A.assemble()

    # Compute u
    u = PETSc.Vec().createSeq(n)

    # Initialize ksp solver.
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    # Solve!
    one = PETSc.Vec().createSeq(n)
    one[:] = np.ones(n)
    ksp.solve(one, u)

    # pass to np.array
    u = u.getArray()

    # theoretical error correction
    umin = 1/np.max(p)
    u = np.maximum(umin, u)

    pmin = np.min(p)
    if pmin > 0:
        umax = 1/pmin
        u = np.minimum(umax, u)

    return u


# make a self-adjoint Hamiltonian
# -\Delta + p
# with Dirichlet boundary condition
# input:
#   np.array 1D
# rvalue
#   np.array 2D n-by-n
def make_hamiltonian(p: np.ndarray) -> np.ndarray:
    H = np.zeros([len(p), len(p)])
    H[0,:2] = [2+p[0], -1]
    for i in range(1, len(p)-1):
        H[i,i-1:i+2] = [-1, 2+p[i], -1]
    H[-1,-2:] = [-1, 2+p[-1]]

    return H

# compute the number of eigenvalues of the Hamiltonian less or equal to 0
# input:
#   H is an n x n np.array
#   overwrite_a: bool (compute in place?)
# rvalue:
#   np.float64
def eigenvalue_count(H: np.ndarray, overwrite_a=True) -> int:
    lu, d, perm = ldl(H, overwrite_a=overwrite_a)
    n = d.shape[0]

    res = i = 0
    while i < n-1:
        if abs(d[i][i+1]) < TINY:
            res += d[i][i] <= 0
            i += 1
        else:
            det = d[i][i] * d[i+1][i+1] - d[i][i+1]**2
            if det < -TINY:
                res += 1
            elif -TINY <= det < TINY:
                res += (d[i][i] + d[i+1][i+1]) <= 0
            else:
                res += 2*((d[i][i] + d[i+1][i+1]) <= 0)
            i += 2

    if i == n-1:
        res += d[i][i] <= 0

    return res

############################################
# Data generation
############################################

# produce a data point as defined in log.md
# saved with save_data_set
# arg is assumed to be supplied if ptype is not ""
# input:
#   domain_size: int
#   domain_type: str; one of discrete or continuous
#   batch_size: int; number of nve
#   ptype: str; type of potential
#   arg: double; used for gen_potential
#   pmax: double; 0.1 <= pmax <= 10, any other number will be disregarded and a new pmax is generated within [0.1,10]
#   seed: int; seed is for generating the potential! not for selecting hyperparameters
def gen_data(domain_size: int, domain_type="discrete", batch_size=100, ptype="", arg=0.0, pmax=0.0, seed=0) -> tuple:
    # if no type is specified,
    # randomly generate one
    # 1/4 probability for ech uniform, bernoulli, and speckle
    # 1/4 for the rest
    if not ptype:
        idx = np.random.randint(0,4)
        print(idx)
        if idx != 3:
            ptype = potential_dict[idx]
        else:
            ptype = choice(potential_dict[3:])

    # only bernoulli and power uses an arg
    if not arg:
        if ptype == "bernoulli": # p in [0.1, 0.9]
            arg = 0.8*np.random.rand()+0.1
        elif ptype == "power": # x^a, a in [0.5, 2]
            arg = 1.5*np.random.rand()+0.5

    # pmax in [0.1, 10] if p is bounded,
    # else pmax = 1
    # default pmax=0.0 automatically < 0.1, so modification is made below
    if ptype in bounded_potential and pmax < 0.1 or pmax > 10:
        pmax = 9.9*np.random.rand()+0.1
    if ptype not in bounded_potential:
        pmax = 1.0


    # From here on ptype, arg, and pmax are all defined

    # pinfo
    pinfo = ""
    pinfo += "ptype: " + ptype +"\n"
    pinfo += "number of eigenvalue counts: " + str(batch_size) + "\n";
    pinfo += "domain_size: " + str(domain_size) + "\n"
    pinfo += "pmax/pmean: " + str(pmax) + "\n"
    pinfo += "arg (0 = unspecified): " + str(arg) + "\n"
    pinfo += "seed (0 = unspecified): " + str(seed) + "\n"
    pinfo += "generator: PETSc for the landscape and scipy.linalg.ldl for eigenvalue counting\n"


    # compute pds
    p = pmax* gen_potential(ptype, domain_size, arg, seed)
    W = 1/compute_u(p)
    p = p.reshape(domain_size, 1)
    W = W.reshape(domain_size, 1)
    pds = np.concatenate([p,W], axis = 1)
    #print(all(pds[i,0] == p[i] for i in range(domain_size)))

    # compute nve
    H = make_hamiltonian(p)
    nve = np.empty([batch_size, 2])
    for i in range(batch_size):
        E = np.random.rand()
        idos = eigenvalue_count(H - np.eye(domain_size)*E)
        nve[i,:] = [E,idos]


    return (pinfo, pds, nve)

def datagen(domain_size, number_of_batches, batch_size=100, data_folder_path="data"):
    for i in range(number_of_batches):
        print("Generating batch {}...".format(i))
        pinfo, pds, nve = gen_data(domain_size, batch_size=batch_size)
        print(pinfo.split('\n')[0])
        file_name = save_data_set(pinfo, pds, nve, data_folder_path, "discrete_" + str(domain_size))
        print("Batch {} saved as ".format(i) + file_name +"\n")


############################################
# Testing
############################################
datagen(10000, 2, 100)



#rint(pinfo)
#print(pds.shape)
#print(nve)
#
# # Plot
# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# plt.scatter(nve[:,0], nve[:,1], color='blue')
# plt.show()



#
# np.random.seed(12)
# n = 200
# p = np.random.rand(n)
# H = make_hamiltonian(p)
# u = compute_u(p)
# print(eigenvalue_count(H-np.eye(n)*3))
#
