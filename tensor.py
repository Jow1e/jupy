from .functional import *
from .activations import *
from .criterions import *
from .linear import *

parse_fn = {
	"add": (add_forward, add_backward),
	"sub": (sub_forward, sub_backward),
	"mul": (mul_forward, mul_backward),
	"neg": (neg_forward, neg_backward),
	"div": (div_forward, div_backward),
	"pow": (pow_forward, pow_backward),
	"log": (log_forward, log_backward),
	"square": (square_forward, square_backward),
	
	"sigmoid": (sigmoid_forward, sigmoid_backward),
	"prelu": (prelu_forward, prelu_backward),
	
	"mse": (mse_forward, mse_backward),
	"cross_entropy": (cross_entropy_forward, cross_entropy_backward),
	
	"linear": (linear_forward, linear_backward),
	
}


def apply_forward(inputs: list, name_fn: str):
	requires_grad = False
	
	fwd_fn, bwd_fn = parse_fn[name_fn]
	
	for idx in range(len(inputs)):
		if not isinstance(inputs[idx], Tensor):
			inputs[idx] = Tensor(inputs[idx], requires_grad=False)
		
		inputs[idx].node.num_out += 1
		requires_grad |= inputs[idx].requires_grad
	
	numpy_variables = [tensor.data for tensor in inputs]
	data, cache = fwd_fn(*numpy_variables)
	
	out = Tensor(data, inputs[0].dtype, requires_grad)
	grad_node = out.node
	
	grad_node.cache = cache
	grad_node.variables = inputs
	grad_node.numpy_variables = numpy_variables
	grad_node.backward_fn = bwd_fn
	
	return out


class GradNode(object):
	def __init__(self):
		self.grad = None
		self.cache = None  # cache for speed-ups
		self.backward_fn = None  # derivative function
		self.variables = None  # list[Tensor...]
		self.numpy_variables = None  # list[Numpy...]
		self.num_out = 0
	
	def backward(self):
		if self.grad is None:
			raise ValueError("GradNode has grad = None")
		
		if self.backward_fn is None:
			return
		
		grads = self.backward_fn(self.grad, self.numpy_variables, self.cache)
		
		for tensor, grad in zip(self.variables, grads):
			node = tensor.node
			node.num_out -= 1
			node.grad = grad if (node.grad is None) else (node.grad + grad)
			
			if node.num_out == 0 and tensor.requires_grad:
				node.backward()


class Tensor(object):
	def __init__(self, data, dtype=float, requires_grad=False):
		self.data = np.asarray(data, dtype)
		self.dtype = dtype
		self.requires_grad = requires_grad
		self.node = GradNode()
	
	@property
	def grad(self):
		return self.node.grad
	
	def backward(self):
		self.node.grad = 1
		self.node.backward()
	
	def numpy(self):
		return self.data
	
	def __add__(self, other):
		return apply_forward([self, other], "add")
	
	def __sub__(self, other):
		return apply_forward([self, other], "sub")
	
	def __mul__(self, other):
		return apply_forward([self, other], "mul")
	
	def __neg__(self):
		return apply_forward([self], "neg")
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def __rsub__(self, other):
		return apply_forward([other, self], "sub")
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __str__(self):
		return self.data.__str__()
	
	def __pow__(self, power, modulo=None):
		return apply_forward([self, power], "pow")
	
	def __rpow__(self, other):
		return apply_forward([other, self], "pow")
