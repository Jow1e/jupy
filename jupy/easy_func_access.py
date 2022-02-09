from .tensor import apply_forward


def square(z):
	return apply_forward([z], "square")


def log(z):
	return apply_forward([z], "log")


def mse(y_hat, y):
	return apply_forward([y_hat, y], "mse")


def cross_entropy(y_hat, y):
	return apply_forward([y_hat, y], "cross_entropy")


def linear(z, weight, bias):
	return apply_forward([z, weight, bias], "linear")


def sigmoid(z):
	return apply_forward([z], "sigmoid")


def prelu(z, slope):
	return apply_forward([z, slope], "prelu")
