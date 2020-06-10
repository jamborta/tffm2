import tensorflow as tf  # type: ignore
from . import utils
import math
import numpy as np  # type: ignore
from typing import Callable, Union, Optional


class TFFMCore(object):
	"""This class implements underlying routines about creating computational graph.

	Its required `n_features` to be set at graph building time.


	Parameters
	----------
	loss_function : loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
		Loss function.
		Take 2 tf.Tensor: outputs and targets and should return tf.Tensor of loss
		See examples: .utils.loss_mse, .utils.loss_logistic

	order : int
		Order of corresponding polynomial model.
		All interaction from bias and linear to order will be included.

	rank : int
		Number of factors in low-rank appoximation.
		This value is shared across different orders of interaction.

	optimizer : tf.train.Optimizer
		Optimization method used for training

	reg : float
		Strength of L2 regularization

	init_std : float
		Amplitude of random initialization

	use_diag : bool
		Use diagonal elements of weights matrix or not.
		In the other words, should terms like x^2 be included.
		Ofter reffered as a "Polynomial Network".
		Default value (False) corresponds to FM.

	reweight_reg : bool
		Use frequency of features as weights for regularization or not.
		Should be usefull for very sparse data and/or small batches

	seed : Optional[int]
		Random seed used at graph creating time

	n_features : Optional[int]
		Number of features used in this dataset.
		if not set, it is inferred during the first call of fit() method.

	Attributes
	----------
	b : tf.Variable
		Bias term.

	w : List[tf.Variable]
		Array of underlying representations.
		First element will have shape [n_features, 1],
		all the others -- [n_features, rank].

	Notes
	-----
	Parameter `rank` is shared across all orders of interactions (except bias and
	linear parts).

	References
	----------
	Steffen Rendle, Factorization Machines
		http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

	"""

	def __init__(self,
				 loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
				 order: int,
				 rank: int,
				 optimizer: tf.optimizers,
				 reg: float,
				 init_std: float,
				 use_diag: bool,
				 reweight_reg: bool,
				 seed: Optional[int],
				 n_features: Optional[int]):
		self.order = order
		self.rank = rank
		self.use_diag = use_diag
		self.optimizer = optimizer
		self.reg = reg
		self.reweight_reg = reweight_reg
		self.init_std = init_std
		self.seed = seed
		self.n_features = n_features
		self.graph = None
		self.loss_function = loss_function

	def set_num_features(self, n_features: int) -> None:
		self.n_features = n_features

	def init_weights(self) -> None:
		self.w = [None] * self.order
		for i in range(1, self.order + 1):
			r = self.rank
			if i == 1:
				r = 1
			rnd_weights = tf.random.uniform([self.n_features, r], -self.init_std, self.init_std)
			self.w[i - 1] = tf.Variable(rnd_weights, trainable=True, name='embedding_' + str(i))
		self.b = tf.Variable(self.init_std, trainable=True, name='bias')
		self.step = tf.Variable(1, name='step', dtype=tf.int64)
		self.regularization = tf.Variable(0.0, name='regularization')
		tf.summary.scalar('bias', self.b, step=self.step)

	@tf.function
	def __call__(self, train_x: tf.Tensor) -> tf.Tensor:
		with tf.name_scope('linear_part'):
			contribution = tf.matmul(train_x, self.w[0])
		y_pred = self.b + contribution
		for i in range(2, self.order + 1):
			with tf.name_scope('order_{}'.format(i)):
				raw_dot = tf.matmul(train_x, self.w[i - 1])
				dot = tf.pow(raw_dot, i)
				if self.use_diag:
					contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
					contribution /= 2.0 ** (i - 1)
				else:
					initialization_shape = tf.shape(dot)
					for in_pows, out_pows, coef in utils.powers_and_coefs(i):
						product_of_pows = tf.ones(initialization_shape)
						for pow_idx in range(len(in_pows)):
							x_pow = tf.pow(train_x, in_pows[pow_idx])
							w_pow = tf.pow(self.w[i - 1], in_pows[pow_idx])
							pmm = tf.matmul(x_pow, w_pow)
							product_of_pows *= tf.pow(pmm, out_pows[pow_idx])
						dot -= coef * product_of_pows
					contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
					contribution = contribution / float(math.factorial(i))
			y_pred = y_pred + contribution

		with tf.name_scope('regularization'):
			with tf.name_scope('reweights'):
				if self.reweight_reg:
					counts = tf.math.count_nonzero(train_x, axis=0, keepdims=True)
					sqrt_counts = tf.transpose(tf.sqrt(tf.cast(counts, np.float32)))
				else:
					sqrt_counts = tf.ones_like(self.w[0])
				reweights = sqrt_counts / tf.reduce_sum(sqrt_counts)
			for order in range(1, self.order + 1):
				node_name = 'regularization_penalty_' + str(order)
				norm = tf.reduce_mean(tf.pow(self.w[order - 1] * reweights, 2), name=node_name)
				tf.summary.scalar('penalty_W_{}'.format(order), norm, step=self.step)
				self.regularization.assign_add(norm)
				tf.summary.scalar('regularization_penalty', self.regularization, step=self.step)
		return y_pred

	@tf.function
	def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, w: tf.Tensor):
		with tf.name_scope('loss'):
			loss = self.loss_function(y_true, y_pred) * w
			reduced_loss = tf.reduce_mean(loss)
		target = reduced_loss + self.reg * self.regularization
		checked_target = tf.debugging.assert_all_finite(
			target,
			message='NaN or Inf in target value',
			name='target')
		return reduced_loss, checked_target

	def train(self, X, y, w):
		with tf.GradientTape() as t:
			_, current_loss = self.loss(y, self(X), w)
		vars = self.w + [self.b]
		grads = t.gradient(current_loss, vars)
		self.optimizer.apply_gradients(zip(grads, vars))
		self.step.assign_add(1)
		return current_loss
