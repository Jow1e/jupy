import numpy as np


class SGD(object):
	def __init__(self, parameters, lr=1e-3, weight_decay=0):
		self.lr = lr
		self.weight_decay = weight_decay
		self.parameters = parameters
	
	def reset_grad(self):
		for param in self.parameters:
			param.node.grad = None
	
	def step(self):
		for param in self.parameters:
			param.data *= (1 - self.weight_decay)
			param.data -= self.lr * param.grad


class AdamW(object):
	def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
		self.lr = lr
		self.beta1, self.beta2 = betas
		self.weight_decay = weight_decay
		self.eps = eps
		self.iters = 0
		
		self.parameters = parameters
		self.momentum1 = [np.zeros_like(param) for param in parameters]
		self.momentum2 = self.momentum1.copy()
	
	def reset_grad(self):
		for param in self.parameters:
			param.node.grad = None
	
	def step(self):
		self.iters += 1
		
		for index in range(len(self.parameters)):
			g = self.parameters[index].grad
			m = self.momentum1[index]
			v = self.momentum2[index]
			
			# calculate first and second moments (m, v)
			# maybe use in-place operations for speed?
			m = self.beta1 * m + (1 - self.beta1) * g
			v = self.beta2 * v + (1 - self.beta2) * np.square(g)
			
			# bias correction (not to make it go toward zero)
			m_hat = m / (1 - self.beta1 ** self.iters)
			v_hat = v / (1 - self.beta2 ** self.iters)
			
			# weight regularization
			self.parameters[index].data *= (1 - self.lr * self.weight_decay)
			
			# optimizer step
			self.parameters[index].data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
			
			# update momentum
			self.momentum1[index] = m
			self.momentum2[index] = v
