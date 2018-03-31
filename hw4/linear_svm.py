import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  reg = np.dot(theta, theta) / (2.0 * m)

  for (x_, y_) in zip(X, y):
    v = y_ * np.dot(theta, x_)
    J += max(0, 1 - v)
    grad += 0 if v >= 1 else -y_ * x_
  J = J*C/m+reg
  grad = grad/m +theta/m

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  # compute the loss and the gradient
  num_classes = K
  num_train = m


  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(theta)
    correct_class_score = scores[y[i]]
    diff_count = 0.0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        diff_count += 1
        dtheta[:, j] += X[i] # gradient update
        loss += margin
    # gradient update for correct row
    dtheta[:, y[i]] += -diff_count * X[i]

  loss /= num_train
  dtheta /= num_train
  dtheta += reg*dtheta

  loss += 0.5 * reg * np.sum(theta * theta)/m


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################
  
  scores = np.dot(X, theta) # also known as f(x_i, W)

  correct_scores = np.ones(scores.shape).T * scores[np.arange(0, scores.shape[0]),y]
  deltas = np.ones(scores.shape)
  L = scores - correct_scores.T + deltas

  L[L < 0] = 0
  L[np.arange(0, scores.shape[0]),y] = 0 # Don't count y_i
  loss = np.sum(L)

  # Average over number of training examples
  num_train = X.shape[0]
  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(theta * theta)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad = np.zeros(scores.shape)

  L = scores - correct_scores.T + deltas

  L[L < 0] = 0
  L[L > 0] = 1
  L[np.arange(0, scores.shape[0]), y] = 0  # Don't count y_i
  L[np.arange(0, scores.shape[0]), y] = -1 * np.sum(L, axis=1 )
  dtheta = np.dot( X.T,L)

  # Average over number of training examples
  num_train = X.shape[0]
  dtheta /= num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
