import numpy as np

# make_speckle
#   input:
#       domain_size: int
#       sigma: float
#       vmax: float
#       nc: int number of bins
#   rvalue:
#       np.array size: domain_size
def make_single_speckle(domain_size, sigma, vmax):
    # courtesy of Doug Arnold

    #Sample two independent random vectors X,Y ~ N(0,1) with spatial correlation
    #cov(X,Y) = sinc(|n-m|/(2*sigma))
    covmat = np.fromfunction(lambda n, m: np.sinc((n - m)/sigma), (domain_size, domain_size))

    #print(covmat.shape)
    X = np.random.multivariate_normal(mean=np.zeros(domain_size), cov=covmat)
    Y = np.random.multivariate_normal(mean=np.zeros(domain_size), cov=covmat)

    #print(X.shape)
    #Create random vector V with PDF 1/Vmax*exp(-x/Vmax) using X and Y
    V = vmax*(X**2 + Y**2)/2.
    return V

# all int ranges are exclusive!
def make_speckle(batch_size, domain_size, sigma_range, vmax_range):
    res = np.empty((batch_size, domain_size))
    for i in range(batch_size):
        sigma = (sigma_range[1]-sigma_range[0])*np.random.random()+sigma_range[0]
        vmax = (vmax_range[1]-vmax_range[0])*np.random.random()+vmax_range[0]

        res[i] = make_single_speckle(domain_size, sigma, vmax)

    return res

x = make_speckle(1, 10000, [0.1,10], [0.1,10])
