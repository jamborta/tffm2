"""Implementation of an arbitrary order Factorization Machines."""

import numpy as np  # type: ignore
from .base import TFFMBaseModel
from .utils import loss_logistic, loss_mse, loss_bpr
import tensorflow as tf  # type: ignore
from typing import Union, Optional, Callable, Iterable


class TFFMClassifier(TFFMBaseModel):
	"""Factorization Machine (aka FM).

	This class implements L2-regularized arbitrary order FM model with logistic
	loss and gradient-based optimization.

	Only binary classification with 0/1 labels supported.

	See TFFMBaseModel and TFFMCore docs for details about parameters.
	"""

	def __init__(self,
				 loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
				 order: int = 2,
				 rank: int = 2,
				 optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
				 reg: float = 0.0,
				 init_std: float = 0.01,
				 use_diag: bool = False,
				 reweight_reg: bool = False,
				 seed: Optional[int] = None,
				 n_features: Optional[int] = None,
				 n_epochs: int = 100,
				 batch_size: Optional[int] = None,
				 shuffle_size: int = 1000,
				 checkpoint_dir: Optional[str] = None,
				 summary_dir: str = ".",
				 log_dir: Optional[str] = None,
				 verbose: int = 0,
				 sample_weight: Union[None, str, np.ndarray] = None,
				 pos_class_weight: float = None,
				 eval_step: int = 100):

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
			n_features=n_features,
			n_epochs=n_epochs,
			batch_size=batch_size,
			shuffle_size=shuffle_size,
			checkpoint_dir=checkpoint_dir,
			summary_dir=summary_dir,
			log_dir=log_dir,
			verbose=verbose,
			eval_step=eval_step
		)

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

	def fit(self, X_train: Union[np.ndarray, tf.data.Dataset],
			X_val: Union[np.ndarray, tf.data.Dataset, None] = None,
			y_train: Optional[np.ndarray] = None,
			y_val: Optional[np.ndarray] = None,
			sample_weight: Optional[np.array] = None, pos_class_weight: Optional[float] = None,
			n_epochs: Optional[int] = None,
			show_progress: bool = False):
		# TODO: add validation dataset
		if isinstance(X_train, np.ndarray):
			# preprocess Y: suppose input {0, 1}, but internally will use {-1, 1} labels instead
			if not (set(y_train) == {0, 1}):
				raise ValueError("Input labels must be in set {0,1}.")
			used_y_train = y_train * 2 - 1
			if sample_weight is not None:
				self.sample_weight = sample_weight
			if pos_class_weight is not None:
				self.pos_class_weight = pos_class_weight
			used_w = self._preprocess_sample_weights(self.sample_weight, self.pos_class_weight, used_y_train)
			dataset = self.create_dataset(X_train, used_y_train, used_w)
			self._fit(dataset, n_epochs=n_epochs, show_progress=show_progress)
		elif isinstance(X_train, tf.data.Dataset):
			self._fit(X_train, X_val, n_epochs=n_epochs, show_progress=show_progress)

	def predict(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: Optional[int] = None) -> Iterable:
		"""

		Parameters
		----------
		X: Union[tf.data.Dataset, np.ndarray]
			Samples
		pred_batch_size: Optional[int]
			Batch size for prediction

		Returns
		-------
		Returns predicted values.
		"""
		raw_output = self.decision_function(X, pred_batch_size)
		for r in raw_output:
			yield {**r, "pred": tf.cast(r["pred_raw"] > 0, tf.int8)}

	def predict_proba(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: Optional[int] = None) -> Iterable:
		"""Probability estimates.

		The returned estimates for all 2 classes are ordered by the
		label of classes.

		Parameters
		----------
		X: Union[tf.data.Dataset, np.ndarray]
			Samples
		pred_batch_size : Optional[int]
			Batch size for prediction
		Returns
		-------
		probs: Optional[int]
			Returns the probability of the sample for each class in the model.
		"""
		raw_output = self.decision_function(X, pred_batch_size)
		for r in raw_output:
			yield {**r, "pred_pos": tf.sigmoid(r["pred_raw"]), "pred_neg": 1 - tf.sigmoid(r["pred_raw"])}


class TFFMRegressor(TFFMBaseModel):
	"""Factorization Machine (aka FM).

	This class implements L2-regularized arbitrary order FM model with MSE
	loss and gradient-based optimization.

	Custom loss functions are not supported, mean squared error is always
	used. Any loss function provided in parameters will be overwritten.

	See TFFMBaseModel and TFFMCore docs for details about parameters.
	"""

	def __init__(self, loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
				 order: int = 2,
				 rank: int = 2,
				 optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
				 reg: float = 0,
				 init_std: float = 0.01,
				 use_diag: bool = False,
				 reweight_reg: bool = False,
				 seed: Optional[int] = None,
				 n_features: Optional[int] = None,
				 n_epochs: int = 100,
				 batch_size: Optional[int] = None,
				 shuffle_size: int = 1000,
				 checkpoint_dir: Optional[str] = None,
				 summary_dir: str = ".",
				 log_dir: Optional[str] = None,
				 verbose: int = 0,
				 eval_step: int = 100
				 ):
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
			n_features=n_features,
			n_epochs=n_epochs,
			batch_size=batch_size,
			shuffle_size=shuffle_size,
			checkpoint_dir=checkpoint_dir,
			summary_dir=summary_dir,
			log_dir=log_dir,
			verbose=verbose,
			eval_step=eval_step
		)

	def fit(self, X_train: Union[np.ndarray, tf.data.Dataset],
			X_val: Union[np.ndarray, tf.data.Dataset, None] = None,
			y_train: Optional[np.ndarray] = None,
			y_val: Optional[np.ndarray] = None,
			sample_weight_train: Optional[np.ndarray] = None,
			sample_weight_val: Optional[np.ndarray] = None,
			n_epochs: Optional[int] = None,
			show_progress: bool = False):
		# TODO: add validation dataset
		if isinstance(X_train, np.ndarray):
			sample_weight_train = np.ones_like(y_train) if sample_weight_train is None else sample_weight_train
			sample_weight_val = np.ones_like(y_val) if sample_weight_val is None else sample_weight_val
			assert y_train is not None, "y_train should be defined if X is an array"
			dataset_train = self.create_dataset(X_train, y_train, sample_weight_train)
			if all(v is not None for v in [X_val, y_val]):
				dataset_val = self.create_dataset(X_val, y_val, sample_weight_val, repeat=True)
				self._fit(dataset_train, dataset_val, n_epochs=n_epochs, show_progress=show_progress)
			else:
				self._fit(dataset_train, n_epochs=n_epochs, show_progress=show_progress)
		elif isinstance(X_train, tf.data.Dataset):
			self._fit(X_train, X_val, n_epochs=n_epochs, show_progress=show_progress)

	def predict(self, X: Union[tf.data.Dataset, np.ndarray], pred_batch_size: Optional[int] = None) -> Iterable:
		"""Predict using the FM model

		Parameters
		----------
		X : Union[tf.data.Dataset, np.ndarray]
			Samples.
		pred_batch_size : Optional[int]
			Batch size for prediction
		Returns
		-------
		predictions : array, shape = (n_samples,)
			Returns predicted values.
		"""
		predictions = self.decision_function(X, pred_batch_size)
		return predictions
