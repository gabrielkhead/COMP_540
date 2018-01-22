import numpy as np
import matplotlib.pyplot as plt



def normal_dist(samples, mean, std_dev):

    s = np.random.normal(loc= mean, scale = std_dev, size = samples)    #generate normal values
    count, bins, ignored = plt.hist(s, 30, normed=True)                 #form histogram
    plt.plot(bins, 1 / (std_dev * np.sqrt(2 * np.pi * std_dev ** 2))* np.exp(- (bins - mean) ** 2 / (2 * std_dev ** 2)),
    linewidth = 2, color = 'r')                                         #plot fit of histogram
    plt.ylabel('Normal Distribution')
    plt.show()

def categorical_dist(Distribution, samples):

    s = np.random.multinomial(n = samples, pvals= Distribution, size = 1)
    y = np.arange(len(Distribution))
    plt.bar(y,s[0])
    plt.ylabel('Categorical Distribution')
    plt.show()

def multivarnm_dist(mean_v,Cov_m,samples):
    x, y = np.random.multivariate_normal(mean_v, Cov_m, samples).T
    plt.plot(x, y, 'gx')
    plt.axis('equal')
    plt.ylabel('Multivariate Gaussian Distribution')
    plt.show()

normal_dist(samples =1000, mean =10, std_dev=1)
categorical_dist(Distribution = [0.2,0.4,0.3,0.1],samples=1000)
multivarnm_dist([1,1],[[1,0.5],[1,0.5]],1000)

