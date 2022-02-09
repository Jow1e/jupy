import math
import numpy as np
from .tensor import Tensor, apply_forward


class Sequential(object):
	def __init__(self, *layers):
		self.layers = layers
		
	def __call__(self, input):
		for layer in self.layers:
			input = layer(input)
		
		return input
	
	def parameters(self):
		out = []
		
		for layer in self.layers:
			out.extend(layer.parameters())
			
		return out


class Linear(object):
	def __init__(self, in_nodes, out_nodes):
		# to be sure that Var(output) <= 1 | if Var(input) = 1
		scale = 1 / math.sqrt(in_nodes)
		
		weight = np.random.normal(0, scale, (out_nodes, in_nodes))
		bias   = np.random.normal(0, scale, (out_nodes,))
		
		self.weight = Tensor(weight, requires_grad=True)
		self.bias   = Tensor(bias, requires_grad=True)
		
	def __call__(self, input):
		return apply_forward([input, self.weight, self.bias], "linear")
	
	def parameters(self):
		return [self.weight, self.bias]


class PReLU(object):
	def __init__(self, in_nodes, init_slope=0.15):
		slope = np.full(in_nodes, init_slope)
		self.slope = Tensor(slope, requires_grad=True)
		
	def __call__(self, input):
		return apply_forward([input, self.slope], "prelu")
	
	def parameters(self):
		return [self.slope]
