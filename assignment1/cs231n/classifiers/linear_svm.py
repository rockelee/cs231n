import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # print dW.shape, X.shape

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    indicator = (scores - correct_class_score + 1) > 0
    indicator[y[i]] = False
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += -np.sum(indicator) * X[i]
        continue
      dW[:, j] += indicator[j] * X[i]
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += 2 * reg * W
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)

  margins = np.maximum(0, scores - scores[np.arange(num_train), y].reshape((num_train, 1)) + 1)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins) / num_train
  loss += reg * np.sum(W * W)
  
  indicator = (margins > 0).astype('float64')
  # for i in range(num_train):
    # for j in range(num_classes):
        # if j == y[i]:
          # dW[:, j] += -np.sum(indicator[i]) * X[i]
          
          # dW[:, j] += indicator[i, j] * X[i]
        # else:
          # dW[:, j] += indicator[i, j] * X[i]
        # dW[:, j] += indicator[i, j] * X[i].T
        
  indicator[np.arange(num_train), y] = -np.sum(indicator, axis = 1)
  dW = np.dot(X.T, indicator)
  dW /= num_train
  dW += 2 * reg * W
  
  #print loss
  
  # loss = 0.0
  # for i in range(num_train):
    # margins = np.maximum(0, scores[i, :] - scores[i, y[i]] + 1)
    # margins[y[i]] = 0
    # loss_i = np.sum(margins)
    # loss += loss_i
  # loss /= num_train
  # loss += reg * np.sum(W * W)
  # print loss
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
