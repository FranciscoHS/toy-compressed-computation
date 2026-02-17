import torch
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

@dataclass
class Config:
	"""Configuration class for model and training"""
	# Model Parameters
	n_features: int = 100
	d_embed: int = 1000
	d_mlp: int = 50

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
		# The embedding vectors are random vectors; not trainable
		# Initialized with Xavier normal distribution; apparently good against vanishing gradients
		# See https://proceedings.mlr.press/v9/glorot10a
		self.W_E = xavier_normal((d_embed, n_features), n_features, d_embed)
		self.W_in = torch.randn(d_mlp, d_embed)
		self.W_in.requires_grad_()
		self.W_out = torch.rand(d_embed, d_mlp)
		self.W_out.requires_grad_()

	@staticmethod
	def activation(x):
		# Just a ReLU
		return torch.maximum(torch.zeros(1), x)	

	def forward(self, x):
		# @ is shorthand for matmul in torch starting with Python 3.5
		# we need to transpose to align dimensions and keep batch size first dimension
		res = x @ self.W_E.T
		pre_act = res @ self.W_in.T
		post_act = self.activation(pre_act)
		mlp_out = post_act @ self.W_out.T
		res = res + mlp_out
		# transposing twice is identity
		return res @ self.W_E


class UniformFeatureDataset():
	"""Uniformly distributed features in [-1, 1]. The labels are ReLU(feature) + feature
	"""

	def __init__(self, config: Config):
		self.n_features = config.n_features

	def generate_batch(self, batch_size):
		batch = 2 * torch.rand(batch_size, self.n_features) - 1
		labels = SimpleMLP.activation(batch) + batch
		return batch, labels


def xavier_normal(shape: Sequence[int], n_inputs: int, n_outputs: int) -> torch.Tensor:
	std = np.sqrt(2. / (n_inputs + n_outputs))
	return torch.distributions.Normal(loc=0., scale=std).sample(shape)


def train(network: SimpleMLP, config: Config) -> SimpleMLP:	
	optimizer = torch.optim.AdamW(params=[network.W_in, network.W_out], lr=config.lr, weight_decay=0.01)

	dataset = UniformFeatureDataset(config)
	for step in range(config.steps):
		batch, labels = dataset.generate_batch(config.batch_size)
		optimizer.zero_grad()  # clears gradients
		outputs = network.forward(batch)
		loss = torch.mean((labels - outputs) ** 2)
		loss.backward()
		optimizer.step()

		if (step + 1) % config.print_feq == 0 or step == config.steps - 1:
			print("Loss {} at step {}".format(loss.item(), step)) 

	return network

def evaluate(network: SimpleMLP, dataset: UniformFeatureDataset, batch_size: int=10000):
	with torch.no_grad():
		batch, labels = dataset.generate_batch(batch_size)
		outputs = model.forward(batch)
		loss = torch.mean((labels - outputs) ** 2).item()
	return loss


torch.manual_seed(42)
np.random.seed(42)
config = Config()
model_path = Path('model.pkl')
if not model_path.exists():
	model = SimpleMLP(config)
	model = train(network=model, config=config)
	with open(model_path, "wb") as f:
		pickle.dump(model, f)

else:
	with open(model_path, "rb") as f:
		model = pickle.load(f)
dataset = UniformFeatureDataset(config)
print("Trained model loss: ", evaluate(model, dataset))
