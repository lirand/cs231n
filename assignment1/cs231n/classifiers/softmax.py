from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)

    for i in range(num_train):
        fexp = np.exp(scores[i] - np.max(scores[i]))
        softmax = fexp / np.sum(fexp)
        loss += (-1) * np.log(softmax[y[i]])

        # Soft max derivative:
        #pk * X            if i != j
        #pk * (X - 1)      if i = j

        for j in range(num_classes):
            dW[:, j] += X[i] * softmax[j]
        dW[:, y[i]] -= X[i]

    loss /=  num_train
    dW /= num_train

    # Regularization
#    loss += reg * np.sum(W * W)
#    dW += reg * 2 * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1, keepdims=True)

    fexp = np.exp(scores)
    softmax = fexp / np.sum(fexp, axis=1).reshape(-1, 1)


    loss_per_example = (-1) * np.log(softmax[np.arange(num_train), y])
    loss = np.sum(loss_per_example)

    softmax[np.arange(num_train), y] -= 1

    dW = (X.T).dot(softmax)

    loss = loss / num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

