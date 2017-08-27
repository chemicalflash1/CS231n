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
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# Compute the softmax loss and its gradient using explicit loops.           #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in range(num_train):
		scores = X[i].dot(W)
		scores_normalized = scores - np.max(scores)
		scores_normalized_exp = np.exp(scores_normalized)
		probabilities = scores_normalized_exp / sum(scores_normalized_exp)

		correct_class_probability = probabilities[y[i]]
		loss += - np.log(correct_class_probability)

		#gradient
		dscores = probabilities
		dscores[y[i]] -= 1
		for j in range(num_classes):
			dW[:, j] += dscores[j] * X[i]

	loss /= num_train
	dW /= num_train

	loss += reg * np.sum(W * W)
	dW += 2 * reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

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
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	num_train = X.shape[0]

	scores = X.dot(W)
	scores_normal = scores - np.max(scores, axis=1)[:, np.newaxis]
	scores_noraml_exp = np.exp(scores_normal)
	probabilities = scores_noraml_exp / np.sum(scores_noraml_exp, axis=1)[:, np.newaxis]
	probabilities_correct_class = probabilities[np.arange(num_train), y]
	loss = sum(- np.log(probabilities_correct_class))

	loss /= num_train
	loss += reg * np.sum(W * W)


	dscores = probabilities
	dscores[np.arange(num_train), y] -= 1
	dW = np.dot(dscores.T, X).T

	dW /= num_train
	dW += 2 * reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW
