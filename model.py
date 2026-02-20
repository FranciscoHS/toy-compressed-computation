import torch
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class Config:
	"""Configuration class for model and training"""
	# Model Parameters
	n_features: int = 2
	d_embed: int = 1000
	d_mlp: int = 1
	p_feature: float = 0.05

	# Training Parameters
	batch_size: int = 2048
	steps: int = 200
	lr: float = 3e-3
	print_feq: int = 20


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
	"""Uniformly distributed features in [-1, 1].
	Each label is ReLU(feature) + feature with probability p_feature, zero otherwise.
	"""

	def __init__(self, config: Config):
		self.n_features = config.n_features
		self.p_feature = config.p_feature

	def generate_batch(self, batch_size):
		batch = 2 * torch.rand(batch_size, self.n_features) - 1
		# rand_like generates a tensor of the same dimensions as its input with uniformly randomly sampled values
		# by comparing to p_feature, only a fraction p_feature will be True, the rest False
		# by then doing elementwise product with our batch, we zero out 1 - p_feature of them
		mask = torch.rand_like(batch) < self.p_feature
		batch = mask * batch
		labels = SimpleMLP.activation(batch) + batch
		return batch, labels


class UniformFeatureDatasetReLU():
	"""Uniformly distributed features in [-1, 1].
	Each label is ReLU(feature) with probability p_feature, zero otherwise.
	"""

	def __init__(self, config: Config):
		self.n_features = config.n_features
		self.p_feature = config.p_feature

	def generate_batch(self, batch_size):
		batch = 2 * torch.rand(batch_size, self.n_features) - 1
		# rand_like generates a tensor of the same dimensions as its input with uniformly randomly sampled values
		# by comparing to p_feature, only a fraction p_feature will be True, the rest False
		# by then doing elementwise product with our batch, we zero out 1 - p_feature of them
		mask = torch.rand_like(batch) < self.p_feature
		batch = mask * batch
		labels = SimpleMLP.activation(batch)
		return batch, labels



class SimpleMLPNoEmbed():
	"""Simple residual network consisting of an MLP without biases with no embeddings.
	Adapted from
	https://www.lesswrong.com/posts/ZxFchCFJFcgysYsT9/compressed-computation-is-probably-not-computation-in
	"""

	def __init__(self, config: Config):
		n_features, d_mlp = config.n_features, config.d_mlp
		self.W_in = torch.randn(d_mlp, n_features)
		self.W_in.requires_grad_()
		self.W_out = torch.rand(n_features, d_mlp)
		self.W_out.requires_grad_()

	@staticmethod
	def activation(x):
		# Just a ReLU
		return torch.maximum(torch.zeros(1), x)	

	def forward(self, x):
		# @ is shorthand for matmul in torch starting with Python 3.5
		# we need to transpose to align dimensions and keep batch size first dimension
		pre_act = x @ self.W_in.T
		post_act = self.activation(pre_act)
		mlp_out = post_act @ self.W_out.T
		return mlp_out



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

def evaluate(model: SimpleMLP, dataset: UniformFeatureDataset, batch_size: int=10000):
	with torch.no_grad():
		batch, labels = dataset.generate_batch(batch_size)
		outputs = model.forward(batch)
		loss = torch.mean((labels - outputs) ** 2).item()
	return loss

def loss_per_feature():
	torch.manual_seed(42)
	np.random.seed(42)
	config = Config()
	config.n_features = 100
	config.p_feature = 1.
	config.d_mlp = 50
	config.steps = 1000
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

	# Look at loss per feature
	batch_size = 10000
	batch, labels = dataset.generate_batch(batch_size)
	with torch.no_grad():
		outputs = model.forward(batch)
		# we here average only over the first dimension, i.e., the batch dimension
		# not over the second dimension, which is the feature dimension
		loss = ((labels - outputs) ** 2).mean(dim=0)

	plt.plot(loss)
	plt.title('MSE per feature')
	plt.xlabel('Feature')
	plt.ylabel('MSE')
	plt.savefig('loss_per_feature_mlp_embed_100_features_dense.png')

for i in range(1, 100):
	p_feature = float(i) / 100
	torch.manual_seed(42)
	np.random.seed(42)
	config = Config()
	config.p_feature = p_feature
	model_path_name = 'modelnoembed21' + str(p_feature) + '.pkl'
	model_path = Path(model_path_name)
	if not model_path.exists():
		model = SimpleMLPNoEmbed(config)
		model = train(network=model, config=config)
		with open(model_path, "wb") as f:
			pickle.dump(model, f)
	else:
		with open(model_path, "rb") as f:
			model = pickle.load(f)
	dataset = UniformFeatureDatasetReLU(config)
	# print("Feature prob: {} Trained model loss: {}".format(p_feature, evaluate(model, dataset)))

# Look at loss as sparsity varies
loss_per_sparsity = np.zeros(99)
loss_spread = np.zeros(99)
for i in range(1, 100):
	p_feature = float(i) / 100
	torch.manual_seed(42)
	np.random.seed(42)
	config = Config()
	config.p_feature = p_feature
	model_path_name = 'modelnoembed21' + str(p_feature) + '.pkl'
	model_path = Path(model_path_name)
	if not model_path.exists():
		model = SimpleMLPNoEmbed(config)
		model = train(network=model, config=config)
		with open(model_path, "wb") as f:
			pickle.dump(model, f)
	else:
		with open(model_path, "rb") as f:
			model = pickle.load(f)
	dataset = UniformFeatureDatasetReLU(config)
	batch_size = 1000
	batch, labels = dataset.generate_batch(batch_size)
	with torch.no_grad():
		outputs = model.forward(batch)
		loss = torch.mean((labels - outputs) ** 2).item()
		loss_per_feature = ((labels - outputs) ** 2).mean(dim=0)
	loss_spread[i - 1] = np.abs(loss_per_feature[0].item() - loss_per_feature[1].item()) / torch.mean(loss_per_feature)
	loss_per_sparsity[i - 1] = loss

plt.scatter([x / 100 for x in range(1, 100)], loss_per_sparsity)
plt.title('MSE versus Sparsity')
plt.xlabel('Feature Probability')
plt.ylabel('MSE')
plt.savefig('MSE_Sparsity_Two_Features_One_Neuron.png')

plt.clf()
plt.scatter([x / 100 for x in range(1, 100)], loss_spread)
plt.title('MSE Spread vs Sparsity')
plt.xlabel('Feature Probability')
plt.ylabel('Normalized MSE Difference')
plt.savefig('MSE_Spread_Sparsity_Two_Features_One_Neuron.png')


"""
# Look at loss per feature
batch_size = 1000
batch, labels = dataset.generate_batch(batch_size)
with torch.no_grad():
	outputs = model.forward(batch)
	# we here average only over the first dimension, i.e., the batch dimension
	# not over the second dimension, which is the feature dimension
	loss = ((labels - outputs) ** 2).mean(dim=0)

plt.scatter(range(config.n_features), loss)
plt.title('MSE per feature')
plt.xlabel('Feature')
plt.ylabel('MSE')
plt.savefig('loss_per_feature_mlp_no_embed_two_features.png')
"""