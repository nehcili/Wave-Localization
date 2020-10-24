# 2018-10-05  Douglas N Arnold
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import diric
import os
from datetime import datetime
"""
Generate and store periodic speckle patterns of specified length, scale,
and correlation.

The correlation length is always of the form nc/q with q an odd integer, so
if another correlation length is specified it is rounded to this form.

Parameters are:

nc:     vector length
vbar:   mean at each point
q:      odd integer setting the correlation length
sigma:  correlation length = (x1 - x0)/q
seed:   random seed

The distribution of the random variables produced V_n (0 <= n < nc)
is exponential with PDF  vbar exp(-x/vbar), and correlation

corr(V_n, V_m) = [diric(2 pi (n - m)/nc, q)]^2
               = {sin(pi q (n - m)/nc)/[q sin(pi (n - m)/nc)]}^2

The correlation length is given by the first positive zero of the correlation
function: sigma = nc/q.
"""

def correlated_normals(n, w, m):
    """
    N.B. This is only used by speckle() if use_dft=True (the default).

    Returns a pair of real arrays of shape (m, n), for which of each the 2m rows
    (m in the first array and m in the second) is an independent realization
    of a correlated Gaussian field, all with mean 0.  More specifically,
    denoting by v_k, k = 0, 1, ..., n-1, the random field
    sampled by the rows of the two returned arrays, then for each k, v_k is
    normal N(0, 1), and the covariance cov(v_k, v_l) is [d(2 pi (k-l)/n, q)
    where  d(x, q) = sin(q x/2)/(q sin(x/2))  is the Dirichlet function.
    """
    # draw iid N(0, 1) random field of shape (2 m, n)
    a = np.random.standard_normal((2*m, n))
    # combine as a complex iid N(0, 1) random field of shape (m, n)
    b = a[:m, :]  + (1j)*a[m:, :]
    # zero out the columns w + 1, w + 2, ..., n - w - 1
    b[:, w + 1:n - w] = 0.
    # compute the inverse FFT
    z = n/np.sqrt(2*w + 1.) * np.fft.ifft(b)
    return (np.real(z), np.imag(z))

def speckle(nc, w, Vbar=1., size=1, use_dft=True):
    """
    Given positive integer nc and nonnegative integer w with 2 w + 1 < nc / 2,
    returns a real array of shape (size, nc).  Each row is a realization of
    a length nc speckle potential, i.e., of a vector V_n, 0 \le n < nc, with
    each component V_n exponential with scale Vbar (so mean is Vbar and variance is
    Vbar^2), and cov(V_n, V_m) = diric(2 pi (n - m)/nc, 2 w + 1)  (meaning the
    correlation distance is about nc/q).

    If use_dft is True, the correlated Gaussians that are used in the construction
    are computed using the DFT.  Otherwise, numpy's multivariate_normal, which uses
    the SVD, is used.
    """
    # create random variates normal N(0, 1)  with covariance function diric(2 pi x/nc, q)
    if use_dft:
        X, Y = correlated_normals(nc, w, size)
    else:
        covmat = np.fromfunction(lambda n, m: diric(2*pi*(n - m)/nc, 2*w + 1), (nc, nc))
        X = np.random.multivariate_normal(mean=np.zeros(nc), cov=covmat, size=size)
        Y = np.random.multivariate_normal(mean=np.zeros(nc), cov=covmat, size=size)
    # combine to make a correlated exponential field
    return Vbar*(X**2 + Y**2)/2.

starttime = datetime.now()
# set speckle parameters
nc = 100  # length
sigma = 25. # correlation distance (will be adjusted)
Vbar = .5   # scale
seed = 13   # random seed
size = 4 # number of realizations
sfile = 'speckle.txt' # output file name

x0 = 0.0
x1 = float(nc)
L = x1 - x0
np.random.seed(seed)
# set q to odd integer to approximately give correlations distance sigma
q = round(L/sigma)
if q % 2 == 0:
    q += 1
w = (q - 1)//2

print("Run with nc = {}, Vbar = {}, sigma = {}, q = {}, seed = {}".format(nc, Vbar, sigma, q, seed))
V = speckle(nc, w, Vbar, size=size)
np.savetxt(sfile, V, delimiter=',')
endtime = datetime.now()
print("total runtime: {} seconds".format((endtime - starttime).seconds))
