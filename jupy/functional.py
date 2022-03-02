import numpy as np

# this file contains everything for calculating gradient
# e.g. functions (forward, backward)
# "gradient graph" (invisible -> we see only independent nodes as GradNode)


def add_forward(input_1, input_2):
	return input_1 + input_2, None


def add_backward(grad, inputs, cache):
	return grad, grad


def sub_forward(input_1, input_2):
	return input_1 - input_2, None


def sub_backward(grad, inputs, cache):
	return grad, -grad


def mul_forward(input_1, input_2):
	return input_1 * input_2, None


def mul_backward(grad, inputs, cache):
	input_1, input_2 = inputs
	return grad * input_2, grad * input_1


def neg_forward(input):
	return -input, None


def neg_backward(grad, inputs, cache):
	return -grad,


def div_forward(input_1, input_2):
	return input_1 / input_2, None


def div_backward(grad, inputs, cache):
	input_1, input_2 = inputs
	temp = 1 / np.square(input_2)
	return grad * temp * input_2, -grad * input_1 * temp


def square_forward(input):
	return np.square(input), None


def square_backward(grad, inputs, cache):
	return 2 * grad * inputs[0],


def pow_forward(input, power):
	out = input ** power
	return out, out


def pow_backward(grad, inputs, cache):
	input, power = inputs
	return grad * power * (input ** (power - 1)), grad * np.log(power) * cache


def log_forward(input):
	return np.log(input), None


def log_backward(grad, inputs, cache):
	return grad / inputs[0],
