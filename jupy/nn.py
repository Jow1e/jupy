import math
import numpy as np
from .tensor import Tensor, apply_forward


class Module(object):
	def __init__(self):
		self._train = True
	
	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError
	
	def parameters(self):
		return []
	
	def train(self):
		self._train = True
	
	def eval(self):
		self._train = False


class Sequential(Module):
	def __init__(self, *layers):
		super().__init__()
		self.layers = layers
	
	def forward(self, input):
		for layer in self.layers:
			input = layer(input)
		
		return input
	
	def parameters(self):
		out = []
		
		for layer in self.layers:
			out.extend(layer.parameters())
		
		return out


class Linear(Module):
	def __init__(self, in_nodes, out_nodes):
		super().__init__()
		
		# to be sure that Var(output) <= 1 | if Var(input) = 1
		scale = 1 / math.sqrt(in_nodes)
		
		weight = np.random.normal(0, scale, (out_nodes, in_nodes))
		bias = np.random.normal(0, scale, (out_nodes,))
		
		self.weight = Tensor(weight, requires_grad=True)
		self.bias = Tensor(bias, requires_grad=True)
	
	def forward(self, input):
		return apply_forward([input, self.weight, self.bias], "linear")
	
	def parameters(self):
		return [self.weight, self.bias]


class PReLU(Module):
	def __init__(self, in_nodes, init_slope=0.15):
		super().__init__()
		
		slope = np.full(in_nodes, init_slope)
		self.slope = Tensor(slope, requires_grad=True)
	
	def forward(self, input):
		return apply_forward([input, self.slope], "prelu")
	
	def parameters(self):
		return [self.slope]


class BatchNorm(Module):
	def __init__(self, in_features, epsilon=1e-8):
		super().__init__()
		
		self.cache = None
		
		self.in_features = in_features
		self.epsilon = epsilon
		
		self.weight = Tensor(np.ones(in_features), requires_grad=True)
		self.bias = Tensor(np.zeros(in_features), requires_grad=True)
		
		self.sum_squares = np.zeros(in_features)
		self.sum = np.zeros(in_features)
		self.len = 0
		
		self.mean = np.zeros(in_features)
		self.std = np.ones(in_features)
	
	def forward(self, x):
		return apply_forward([x, self.weight, self.bias], fwd_bwd=(self.batchnorm_fwd, self.backward))
	
	def batchnorm_fwd(self, x, weight, bias):
		if self.train:
			x_mean = x.mean(axis=0)
			x_std = x.std(axis=0)
			
			batch_size = x.shape[0]
			
			self.sum_squares += np.sum(x ** 2, axis=0)
			self.sum += np.sum(x, axis=0)
			self.mean = (self.len * self.mean + batch_size * x_mean) / (self.len + batch_size)
			self.len += batch_size
			
			x = (x - x_mean) / (x_std + self.epsilon)
			self.cache = x, x_std, batch_size
		else:
			x = (x - self.mean) / (self.std + self.epsilon)
		
		return weight * x + bias, None
	
	def backward(self, grad, inputs, cache):
		x, x_std, batch_size = self.cache
		
		dx = grad * self.weight
		dx = (batch_size * dx - dx.sum(axis=0) - x * np.sum(x * dx, axis=0)) / (batch_size * x_std + self.epsilon)
		
		dw = np.sum(grad * x, axis=0)
		db = grad.sum(axis=0)
		
		return dx, dw, db
	
	def parameters(self):
		return [self.weight, self.bias]
	
	def eval(self):
		super().eval()
		
		if self.len != 0:
			self.std = np.sqrt((self.sum_squares - 2 * self.mean * self.sum + self.len * (self.mean ** 2)) / self.len)


class Dropout(Module):
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p
		self.factor = 1 / (1 - self.p)
	
	def forward(self, x):
		if self.train:
			mask = np.random.binomial(99, p=self.p, size=x.data.shape) >= self.p * 100
			x = (mask * self.factor) * x
		
		return x
