import numpy as np
import matplotlib.pyplot as plt

#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    ########################################################################
    mu = np.zeros((X.shape[1],))
    sigma = np.ones((X.shape[1],))
    X_norm = np.zeros(X.shape)

    mu = X.mean(0)
    sigma = np.std(X,0)

    for i in xrange(X.shape[1]):
        X_norm[:,i] = (X[:, i ] - mu[i]) / sigma[i]
    ########################################################################
    return X_norm, mu, sigma


#X = np.array([[1,3],[2,2]])
#Y=feature_normalize((X))
#print(Y)