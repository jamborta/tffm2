import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os


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
	batch_size : int, default: -1
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

	def init_basemodel(self, n_epochs=100, batch_size=-1,
					   log_dir=None, session_config=None,
					   verbose=0, seed=None, sample_weight=None,
					   pos_class_weight=None, **core_arguments):
		core_arguments['seed'] = seed
		self.core = TFFMCore(**core_arguments)
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.need_logs = log_dir is not None
		self.log_dir = log_dir
		self.session_config = session_config
		self.verbose = verbose
		self.steps = 0
		self.seed = seed
		self.sample_weight = sample_weight
		self.pos_class_weight = pos_class_weight

	def _fit(self, dataset: tf.data.Dataset, n_epochs: int = None, show_progress: bool = False):
		if self.core.n_features is None:
			self.core.set_num_features(dataset.element_spec['X'].shape[1])
		self.core.init_learnable_params()
		if n_epochs is None:
			n_epochs = self.n_epochs
		for epoch in range(n_epochs):
			for d in dataset:
				current_loss = self.core.loss(self.core(d["X"]), d["y"], d["w"])
				self.core.train(self.core, d["X"], d["y"], d["w"])
				print('Epoch %2d: loss=%2.5f' % (epoch, current_loss))

	def decision_function(self, X, pred_batch_size=None):
		output = []
		if pred_batch_size is None:
			pred_batch_size = self.batch_size

		output = self.core(X)
		distances = np.concatenate(output).reshape(-1)
		# WARNING: be careful with this reshape in case of multiclass
		return distances

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

	def save_state(self, path):
		self.core.saver.save(self.session, path)

	def load_state(self, path):
		if self.core.graph is None:
			self.core.build_graph()
			self.initialize_session()
		self.core.saver.restore(self.session, path)

	def destroy(self):
		"""Terminates session and destroyes graph."""
		self.core.graph = None
