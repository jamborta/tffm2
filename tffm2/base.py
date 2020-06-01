import tensorflow as tf  # type: ignore
from .core import TFFMCore
from sklearn.base import BaseEstimator  # type: ignore
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
from typing import Union, Callable, Optional


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

	log_dir : str or None, default: None
		Path for storing model stats during training. Used only if is not None.
		WARNING: If such directory already exists, it will be removed!
		You can use TensorBoard to visualize the stats:
		`tensorboard --logdir={log_dir}`

	session_config : tf.ConfigProto or None, default: None
		Additional setting passed to tf.Session object.
		Useful for CPU/GPU switching, setting number of threads and so on,
		`tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

	verbose : int, default: 0
		Level of verbosity.
		Set 1 for tensorboard info only and 2 for additional stats every epoch.

	kwargs : dict, default: {}
		Arguments for TFFMCore constructor.
		See TFFMCore's doc for details.

	Attributes
	----------
	core : TFFMCore or None
		Computational graph with internal utils.
		Will be initialized during first call .fit()

	session : tf.Session or None
		Current execution session or None.
		Should be explicitly terminated via calling destroy() method.

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
				 loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Operation],
				 order: int,
				 rank: int,
				 optimizer: tf.optimizers,
				 reg: float,
				 init_std: float,
				 use_diag: bool,
				 reweight_reg: bool,
				 seed: Optional[int],
				 n_epochs: int,
				 batch_size: Optional[int],
				 shuffle_size: int,
				 checkpoint_dir: Optional[str],
				 log_dir: Optional[str],
				 verbose: int):

		self.core = TFFMCore(loss_function=loss_function,
							 order=order,
							 rank=rank,
							 optimizer=optimizer,
							 reg=reg,
							 init_std=init_std,
							 use_diag=use_diag,
							 reweight_reg=reweight_reg,
							 seed=seed)

		self.batch_size = batch_size
		self.shuffle_size = shuffle_size
		self.checkpoint_dir = checkpoint_dir
		self.n_epochs = n_epochs
		self.need_logs = log_dir is not None
		self.log_dir = log_dir
		self.verbose = verbose
		self.steps = 0
		self.seed = seed

	def _fit(self, dataset: tf.data.Dataset, n_epochs: int = None, show_progress: bool = False):
		if self.core.n_features is None:
			self.core.set_num_features(dataset.element_spec['X'].shape[1])
		self.core.init_weights()
		if n_epochs is None:
			n_epochs = self.n_epochs

		if self.checkpoint_dir:
			ckpt = tf.train.Checkpoint(step=self.core.step, optimizer=self.core.optimizer, w=self.core.w, b=self.core.b)
			manager = tf.train.CheckpointManager(ckpt, self.checkpoint_dir, max_to_keep=3)
			ckpt.restore(manager.latest_checkpoint)
			if self.verbose > 1:
				inside_checkpoint = tf.train.list_variables(manager.latest_checkpoint)
				inside_checkpoint_format = "\n".join([f"{a},{b}" for a, b in inside_checkpoint])
				print(f"Checkpoint variables: \n {inside_checkpoint_format}")
			if manager.latest_checkpoint:
				print("Restored from {}".format(manager.latest_checkpoint))
			else:
				print("Initializing from scratch.")

		for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
			for d in dataset:
				current_loss = self.core.loss(self.core(d["X"]), d["y"], d["w"])
				self.core.train(d["X"], d["y"], d["w"])
				if self.checkpoint_dir:
					if int(self.core.step) % 10 == 0:
						save_path = manager.save()
						print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
						print("loss {:1.2f}".format(current_loss.numpy()))
				if self.verbose > 1:
					print('Epoch %2d: loss=%2.5f' % (epoch, current_loss))

	def decision_function(self, X: np.ndarray, pred_batch_size: int = None) -> np.ndarray:
		output = []
		if pred_batch_size is None:
			pred_batch_size = self.batch_size

		dataset = tf.data.Dataset.from_tensor_slices({"X": X})
		if pred_batch_size:
			dataset = dataset.batch(pred_batch_size)
		else:
			dataset = dataset.batch(X.shape[0])
		for d in dataset:
			output.append(self.core(d["X"]))
		distances = np.concatenate(output).reshape(-1)
		# WARNING: be careful with this reshape in case of multiclass
		return distances

	def create_dataset(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tf.data.Dataset:
		dataset = tf.data.Dataset.from_tensor_slices(
			{"X": X, "y": y.astype(np.float32), "w": w.astype(np.float32)}).shuffle(self.shuffle_size)
		if self.batch_size:
			dataset = dataset.batch(self.batch_size)
		else:
			dataset = dataset.batch(X.shape[0])
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
