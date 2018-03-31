import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  # Get shapes
  d, K = theta.shape
  for i in xrange(m):
      inner =  X[i].dot(theta) #prediction i for each K
      inner -= np.max(inner)

      prob = np.exp(inner) / np.sum(np.exp(inner)) #prob of classification for each K
      J += -np.log(prob[y[i]])
      prob[y[i]] -=1

      for j in xrange(K):
          grad[:,j] += X[i,:]*prob[j]

  J /= m
  grad /= m
  J += 0.5 * reg * np.sum(theta*theta)
  grad += reg*theta




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  inner = X.dot(theta)  # prediction i for each K
  inner -= np.max(inner, axis=1, keepdims=True)
  prob = np.exp(inner) / np.sum(np.exp(inner),axis=1, keepdims=True)
  class_prob = prob[range(m),y]

  J = np.sum(-np.log(class_prob))/m
  J += 0.5 * reg * np.sum(theta * theta)
  prob[range(m),y] -=1
  grad = X.T.dot(prob)/m

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
