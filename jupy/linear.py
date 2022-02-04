import numpy as np


def linear_forward(input, weight, bias):
	return np.dot(input, weight.T) + bias, None


def linear_backward(grad, inputs, cache):
	input, weight, bias = inputs
	return np.dot(grad, weight), np.dot(input.T, grad).T, grad.sum(axis=0)
