import numpy as np


def sigmoid_forward(input):
	out = 1 / (1 + np.exp(-input))
	return out, out


def sigmoid_backward(grad, inputs, cache):
	return grad * cache * (1 - cache),


def prelu_forward(input, slope):
	out = np.maximum(input, input * slope)
	return out, None


def prelu_backward(grad, inputs, cache):
	data, slope = inputs
	bool_idx = (data < 0)
	data_grad = grad * (slope * bool_idx + ~bool_idx)
	slope_grad = np.sum(grad * data * bool_idx, axis=0)
	return data_grad, slope_grad
