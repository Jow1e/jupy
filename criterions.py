import numpy as np


def mse_forward(y_hat, y):
	error = y_hat - y
	loss = np.square(error).mean()
	return loss, error


def mse_backward(grad, inputs, cache):
	out = 2 * grad * cache
	return out, -out


def cross_entropy_forward(logits, labels):
	# get batch
	batch_size = len(logits)
	batch_dim = range(batch_size)
	
	# softmax(logit)
	# subtract max to avoid exp overflow
	logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
	logits /= logits.sum(axis=1, keepdims=True)
	
	# calculate loss
	loss = -np.log(logits[batch_dim, labels]).mean()
	
	# calculate gradient (as cache)
	logits[batch_dim, labels] -= 1
	logits /= batch_size
	
	return loss, logits


def cross_entropy_backward(grad, inputs, cache):
	return grad * cache,
