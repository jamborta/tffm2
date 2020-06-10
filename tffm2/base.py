import tensorflow as tf  # type: ignore
from .core import TFFMCore
from sklearn.base import BaseEstimator  # type: ignore
from abc import ABCMeta, abstractmethod
import six
import os
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
from typing import Union, Callable, Optional, Iterable


class TFFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
	"""Base class for FM.
	This class implements L2-regularized arbitrary order FM model.

	It supports arbitrary order of interactions and has linear complexity in the
	number of features (a generalization of the approach described in Lemma 3.1
	in the referenced paper, details will be added soon).

	It can handle both dense and sparse input. Only numpy.array and CSR matrix are
	allowed as inputs; any other input format should be explicitly converted.

	Support logging/visualization with TensorBoard.


	Parameters (for initialization)
	----------
	batch_size : int, default: None
		Number of samples in mini-batches. Shuffled every epoch.
		Use -1 for full gradient (whole training set in each batch).

	n_epoch : int, default: 100
		Default number of epoches.
		It can be overrived by explicitly provided value in fit() method.

	verbose : int, default: 0
		Level of verbosity.
		Set 1 for tensorboard info only and 2 for additional stats every epoch.

	Attributes
	----------
	core : TFFMCore or None
		Computational graph with internal utils.
		Will be initialized during first call .fit()

	steps : int
		Counter of passed lerning epochs, used as step number for writing stats

	n_features : int
		Number of features used in this dataset.
		Inferred during the first call of fit() method.

	intercept : float, shape: [1]
		Intercept (bias) term.

	weights : array of np.array, shape: [order]
		Array of underlying representations.
		First element will have shape [n_features, 1],
		all the others -- [n_features, rank].

	Notes
	-----
	You should explicitly call destroy() method to release resources.
	See TFFMCore's doc for details.
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
				 n_features: Optional[int],
				 n_epochs: int,
				 batch_size: Optional[int],
				 shuffle_size: int,
				 checkpoint_dir: Optional[str],
				 summary_dir: str,
				 log_dir: Optional[str],
				 eval_step: int,
				 verbose: int):

		self.train_writer = tf.summary.create_file_writer(os.path.join(summary_dir, "train"))
		self.validation_writer = tf.summary.create_file_writer(os.path.join(summary_dir, "validation"))

		self.core = TFFMCore(loss_function=loss_function,
							 order=order,
							 rank=rank,
							 optimizer=optimizer,
							 reg=reg,
							 init_std=init_std,
							 use_diag=use_diag,
							 reweight_reg=reweight_reg,
							 seed=seed,
							 n_features=n_features)

		self.batch_size = batch_size
		self.shuffle_size = shuffle_size
		self.checkpoint_dir = checkpoint_dir
		self.n_epochs = n_epochs
		self.need_logs = log_dir is not None
		self.log_dir = log_dir
		self.verbose = verbose
		self.steps = 0
		self.seed = seed
		self.eval_step = eval_step

	def _fit(self, dataset_train: tf.data.Dataset,
			 dataset_val: Optional[tf.data.Dataset] = None,
			 n_epochs: int = None, show_progress: bool = False):
		if self.core.n_features is None:
			n_features = dataset_train.element_spec['X'].shape[1]
			if n_features:
				self.core.set_num_features(n_features)
			else:
				raise Exception("Cannot obtain the number of features from the dataset.")
		self.core.init_weights()
		if n_epochs is None:
			n_epochs = self.n_epochs

		if self.checkpoint_dir:
			ckpt = tf.train.Checkpoint(step=self.core.step, optimizer=self.core.optimizer, w=self.core.w, b=self.core.b,
									   regularization=self.core.regularization)
			manager = tf.train.CheckpointManager(ckpt, self.checkpoint_dir, max_to_keep=3)
			ckpt.restore(manager.latest_checkpoint)
			if manager.latest_checkpoint:
				print("Restored from {}".format(manager.latest_checkpoint))
				if self.verbose > 1:
					inside_checkpoint = tf.train.list_variables(manager.latest_checkpoint)
					inside_checkpoint_format = "\n".join([f"{a},{b}" for a, b in inside_checkpoint])
					print(f"Checkpoint variables: \n {inside_checkpoint_format}")
			else:
				print("Initializing from scratch.")

		training_loss = None
		validation_loss = None
		if dataset_val:
			dataset_val_it = iter(dataset_val)
		pbar = tqdm(range(n_epochs), disable=(not show_progress))
		for epoch in pbar:
			for d in dataset_train:
				pbar.set_description("Training loss: %2.3f, Validation loss: %2.3f, step: %s, epoch: %s" %
									 ((training_loss.numpy() if training_loss else float('inf')),
									  (validation_loss.numpy() if validation_loss else float('inf')),
									  self.core.step.numpy(), epoch))
				self.core.train(d["X"], d["y"], d["w"])
				if int(self.core.step) % self.eval_step == 0:
					training_loss, _ = self.core.loss(d["y"], self.core(d["X"]), d["w"])
					with self.train_writer.as_default():
						tf.summary.scalar("loss", training_loss, step=self.core.step)
					if dataset_val:
						dv = next(dataset_val_it)
						validation_loss, _ = self.core.loss(dv["y"], self.core(dv["X"]), dv["w"])
						with self.validation_writer.as_default():
							tf.summary.scalar("loss", validation_loss, step=self.core.step)
					if self.checkpoint_dir:
						save_path = manager.save()
						if self.verbose > 1:
							print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
					if self.verbose > 1:
						print('Epoch %2d: training loss=%2.3f, validation loss=%2.3f' % (
							epoch, training_loss, validation_loss))

	def decision_function(self, X: Union[tf.data.Dataset, np.ndarray],
						  pred_batch_size: Optional[int] = None) -> Iterable:
		if pred_batch_size is None:
			pred_batch_size = self.batch_size

		if isinstance(X, tf.data.Dataset):
			assert isinstance(X.element_spec, dict), "Expecting a dictionary for the dataset"
			dataset = X
		elif isinstance(X, np.ndarray):
			dataset = tf.data.Dataset.from_tensor_slices({"X": X})
			if pred_batch_size:
				dataset = dataset.batch(pred_batch_size)
			else:
				dataset = dataset.batch(X.shape[0])
		else:
			Exception("Unsupported input type.")

		# Return an iterator (not a dataset) to be able to run predictions of GPUs
		for d in dataset:
			yield {**d, "pred_raw": self.core(d["X"])}

	def create_dataset(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, repeat: bool = False) -> tf.data.Dataset:
		dataset = tf.data.Dataset.from_tensor_slices(
			{"X": X, "y": y.astype(np.float32), "w": w.astype(np.float32)}).shuffle(self.shuffle_size)
		if self.batch_size:
			dataset = dataset.batch(self.batch_size)
		else:
			dataset = dataset.batch(X.shape[0])
		if repeat:
			dataset = dataset.repeat()
		return dataset

	@abstractmethod
	def predict(self, X, pred_batch_size=None):
		"""Predict target values for X."""

	@property
	def intercept(self):
		"""Export bias term from tf.Variable to float."""
		return self.core.b.numpy()

	@property
	def weights(self):
		"""Export underlying weights from tf.Variables to np.arrays."""
		return [x.numpy() for x in self.core.w]
