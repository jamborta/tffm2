This is a TensorFlow 2.0 implementation of an arbitrary order (>=2) Factorization Machine based on paper [Factorization Machines with libFM](http://dl.acm.org/citation.cfm?doid=2168752.2168771).

It supports:
* different (gradient-based) optimization methods
* classification/regression via different loss functions (logistic and mse implemented)

The inference time is linear with respect to the number of features.

Tested on Python3.6


# Dependencies
* [scikit-learn](http://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [tensorflow 2.0+ (tested on 2.2)](https://www.tensorflow.org/)

# Installation
Stable version can be installed via `pip install tffm2`. 

# Usage
The interface is similar to scikit-learn models. To train a 6-order FM model with rank=10 for 100 iterations with learning_rate=0.01 use the following sample
```python
from tffm2 import TFFMClassifier
model = TFFMClassifier(
    order=6,
    rank=10,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    n_epochs=100,
    init_std=0.001
)
model.fit(X_tr, y_tr, show_progress=True)
```

See `example.ipynb` and `gpu_benchmark.ipynb` for more details.

It's highly recommended to read `tffm/core.py` for help.


# Testing
Run ```python test.py``` from the terminal.
 
# Reference
This code is ported from https://github.com/geffy/tffm
```
