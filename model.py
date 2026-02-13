import torch
from dataclasses import dataclass

@dataclass
class Config:
	"""Configuration class for model and training"""
	# Model Parameters
	n_features: int = 50
	d_embed: int = 1000
	d_mlp: int = 100

	# Training Parameters
	batch_size: int = 2048
	steps: int = 1000
	lr: float = 3e-3
	print_feq: int = 50


class SimpleMLP():
	"""Simple residual network consisting of an MLP without biases plus a skip connection.
	Adapted from arXiv:2501.14926
	"""

	def __init__(self, config: Config):
		d_embed, n_features, d_mlp = config.d_embed, config.n_features, config.d_mlp
        # The embedding vectors are random (TODO unit) vectors
		self.W_E = torch.randn(d_embed, n_features)
		self.W_in = torch.randn(d_mlp, d_embed)
		self.W_in.requires_grad_()
		self.W_out = torch.rand(d_embed, d_mlp)
		self.W_out.requires_grad_()

	@staticmethod
	def activation(x):
		# Just a ReLU
		return torch.maximum(0., x)	

	def forward(self, x):
		# @ is shorthand for matmul in torch starting with Python 3.5
		# we need to transpose to align dimensions and keep batch size first dimension
		res = x @ self.W_E.T
		pre_act = res @ self.W_in.T
		post_act = self.activation(pre_act)
		mlp_out = post_act @ self.W_out.T
		res += mlp_out
		# transposing twice is identity
		return res @ self.W_E


class UniformFeatureDataset():
	"""Uniformly distributed features in [-1, 1]. The labels are ReLU(feature) + feature
	"""

	def __init__(self, config: Config):
		self.n_features = config.n_features

	def generate_batch(self, batch_size):
		return 2 * torch.rand(batch_size, n_features) - 1 


def loss(output, target_output):
	# L^2 loss
	return torch.mean((target_output - output) ** 2)

def training():
	pass


config = Config()
x = torch.randn(config.n_features)
my_mlp = SimpleMLP(config)
my_mlp.forward(x)

my_features = UniformFeatureDataset()
print(my_features.generate_batch(batch_size=config.batch_size))