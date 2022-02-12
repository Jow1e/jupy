import numpy as np


def mse_forward(y_hat, y):
	error = y_hat - y
	loss = np.square(error).mean()
	return loss, error


def mse_backward(grad, inputs, cache):
	out = 2 * grad * cache
	return out, -out


def cross_entropy_forward(pred, labels):
	# get batch
	batch_size = len(pred)
	batch_dim = range(batch_size)
	
	# softmax(logit)
	# subtract max to avoid exp overflow
	pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
	pred /= pred.sum(axis=1, keepdims=True)
	
	# calculate loss
	loss = -np.log(pred[batch_dim, labels]).mean()
	
	# calculate gradient (as cache)
	pred[batch_dim, labels] -= 1
	pred /= batch_size
	
	return loss, pred


def cross_entropy_backward(grad, inputs, cache):
	return grad * cache,
