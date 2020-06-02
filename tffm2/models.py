"""Implementation of an arbitrary order Factorization Machines."""

import numpy as np  # type: ignore
from .base import TFFMBaseModel
from .utils import loss_logistic, loss_mse
import tensorflow as tf  # type: ignore
from typing import Union, Optional, Callable


class TFFMClassifier(TFFMBaseModel):
	"""Factorization Machine (aka FM).

	This class implements L2-regularized arbitrary order FM model with logistic
	loss and gradient-based optimization.

	Only binary classification with 0/1 labels supported.

	See TFFMBaseModel and TFFMCore docs for details about parameters.
	"""

	def __init__(self,
				 loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Operation] = None,
				 order: int = 2,
				 rank: int = 2,
				 optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
				 reg: float = 0.0,
				 init_std: float = 0.01,
				 use_diag: bool = False,
				 reweight_reg: bool = False,
				 seed: Optional[int] = None,
				 n_epochs: int = 100,
				 batch_size: Optional[int] = None,
				 shuffle_size: int = 1000,
				 checkpoint_dir: Optional[str] = None,
				 log_dir: Optional[str] = None,
				 verbose: int = 0,
				 sample_weight: Union[None, str, np.ndarray] = None,
				 pos_class_weight: float = None):

		loss = loss_function if loss_function else loss_logistic

		self.sample_weight = sample_weight
		self.pos_class_weight = pos_class_weight

		super().__init__(
			loss_function=loss,
			order=order,
			rank=rank,
			optimizer=optimizer,
			reg=reg,
			init_std=init_std,
			use_diag=use_diag,
			reweight_reg=reweight_reg,
			seed=seed,
			n_epochs=n_epochs,
			batch_size=batch_size,
			shuffle_size=shuffle_size,
			checkpoint_dir=checkpoint_dir,
			log_dir=log_dir,
			verbose=verbose)

	def _preprocess_sample_weights(self, sample_weight: Union[np.ndarray, str, None], pos_class_weight: Optional[float],
								   used_y: np.ndarray):
		assert sample_weight is None or pos_class_weight is None, "sample_weight and pos_class_weight are mutually exclusive parameters"
		used_w = np.ones_like(used_y)
		if sample_weight is None and pos_class_weight is None:
			return used_w
		if type(pos_class_weight) == float:
			used_w[used_y > 0] = pos_class_weight
		elif sample_weight == "balanced":
			pos_rate = np.mean(used_y > 0)
			neg_rate = 1 - pos_rate
			used_w[used_y > 0] = neg_rate / pos_rate
			used_w[used_y < 0] = 1.0
			return used_w
		elif isinstance(sample_weight, np.ndarray) and len(sample_weight.shape) == 1:
			used_w = sample_weight
		else:
			raise ValueError("Unexpected type for sample_weight or pos_class_weight parameters.")

		return used_w

	def fit(self, X: Union[np.ndarray, tf.data.Dataset], y: Optional[np.ndarray],
			sample_weight: Optional[np.array] = None, pos_class_weight: Optional[float] = None,
			n_epochs: Optional[int] = None,
			show_progress: bool = False):
		if isinstance(X, np.ndarray):
			# preprocess Y: suppose input {0, 1}, but internally will use {-1, 1} labels instead
			if not (set(y) == {0, 1}):
				raise ValueError("Input labels must be in set {0,1}.")
			used_y = y * 2 - 1
			if sample_weight is not None:
				self.sample_weight = sample_weight
			if pos_class_weight is not None:
				self.pos_class_weight = pos_class_weight
			used_w = self._preprocess_sample_weights(self.sample_weight, self.pos_class_weight, used_y)
			dataset = self.create_dataset(X, used_y, used_w)
			self._fit(dataset, n_epochs=n_epochs, show_progress=show_progress)
		elif isinstance(X, tf.data.Dataset):
			self._fit(X, n_epochs=n_epochs, show_progress=show_progress)

	def predict(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: Optional[int] = None):
		"""Predict using the FM model

		Parameters
		----------
		X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
			Samples.
		pred_batch_size : int batch size for prediction (default None)

		Returns
		-------
		predictions : array, shape = (n_samples,)
			Returns predicted values.
		"""
		raw_output = self.decision_function(X, pred_batch_size)
		predictions = (raw_output > 0).astype(int)
		return predictions

	def predict_proba(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: Optional[int] = None):
		"""Probability estimates.

		The returned estimates for all 2 classes are ordered by the
		label of classes.

		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
		pred_batch_size : int batch size for prediction (default None)

		Returns
		-------
		probs : array-like, shape = [n_samples, 2]
			Returns the probability of the sample for each class in the model.
		"""
		outputs = self.decision_function(X, pred_batch_size)
		probs_positive = tf.sigmoid(outputs)
		probs_negative = 1 - probs_positive
		probs = np.vstack((probs_negative.T, probs_positive.T))
		return probs.T


class TFFMRegressor(TFFMBaseModel):
	"""Factorization Machine (aka FM).

	This class implements L2-regularized arbitrary order FM model with MSE
	loss and gradient-based optimization.

	Custom loss functions are not supported, mean squared error is always
	used. Any loss function provided in parameters will be overwritten.

	See TFFMBaseModel and TFFMCore docs for details about parameters.
	"""

	def __init__(self, loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Operation] = None,
				 order: int = 2,
				 rank: int = 2,
				 optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
				 reg: int = 0,
				 init_std: float = 0.01,
				 use_diag: bool = False,
				 reweight_reg: bool = False,
				 seed: Optional[int] = None,
				 n_epochs: int = 100,
				 batch_size: Optional[int] = None,
				 shuffle_size: int = 1000,
				 checkpoint_dir: Optional[str] = None,
				 log_dir: Optional[str] = None,
				 verbose: int = 0):
		loss = loss_function if loss_function else loss_mse

		super().__init__(
			loss_function=loss,
			order=order,
			rank=rank,
			optimizer=optimizer,
			reg=reg,
			init_std=init_std,
			use_diag=use_diag,
			reweight_reg=reweight_reg,
			seed=seed,
			n_epochs=n_epochs,
			batch_size=batch_size,
			shuffle_size=shuffle_size,
			checkpoint_dir=checkpoint_dir,
			log_dir=log_dir,
			verbose=verbose)

	def fit(self, X: Union[np.ndarray, tf.data.Dataset], y: Optional[np.ndarray],
			sample_weight: Optional[np.ndarray] = None,
			n_epochs: Optional[int] = None,
			show_progress: bool = False):
		if isinstance(X, np.ndarray):
			sample_weight = np.ones_like(y) if sample_weight is None else sample_weight
			assert y is not None, "y should be defined if X is an array"
			dataset = self.create_dataset(X, y, sample_weight)
			self._fit(dataset, n_epochs=n_epochs, show_progress=show_progress)
		elif isinstance(X, tf.data.Dataset):
			self._fit(X, n_epochs=n_epochs, show_progress=show_progress)

	def predict(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: int = None) -> np.ndarray:
		"""Predict using the FM model

		Parameters
		----------
		X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
			Samples.
		pred_batch_size : int batch size for prediction (default None)

		Returns
		-------
		predictions : array, shape = (n_samples,)
			Returns predicted values.
		"""
		predictions = self.decision_function(X, pred_batch_size)
		return predictions
