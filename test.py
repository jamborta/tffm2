import unittest
import numpy as np
from tffm import TFFMClassifier, TFFMRegressor
from scipy import sparse as sp
import tensorflow as tf


class TestFM(unittest.TestCase):

    def setUp(self):
        # Reproducibility.
        np.random.seed(0)

        n_samples = 20
        n_features = 10

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.binomial(1, 0.5, size=n_samples)

    def classifier(self, use_diag):
        model = TFFMClassifier(
            order=4,
            rank=10,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            n_epochs=0,
            init_std=1.0,
            seed=0,
            use_diag=use_diag
        )
        return model

    def regressor(self, use_diag):
        model = TFFMRegressor(
            order=4,
            rank=10,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            n_epochs=0,
            init_std=1.0,
            seed=0,
            use_diag=use_diag
        )
        return model

    def decision_function_order_4(self, model):

        X = self.X.astype(np.float32)

        model.fit(X, self.y)
        b = model.intercept
        w = model.weights

        desired = self.bruteforce_inference(self.X, w, b, use_diag=model.core.use_diag)

        actual = model.decision_function(X)
        model.destroy()

        np.testing.assert_almost_equal(actual, desired, decimal=4)

    def test_FM_classifier(self):
        self.decision_function_order_4(self.classifier(use_diag=False))

    def test_PN_classifier(self):
        self.decision_function_order_4(self.classifier(use_diag=True))

    def test_FM_regressor(self):
        self.decision_function_order_4(self.regressor(use_diag=False))

    def test_PN_regressor(self):
        self.decision_function_order_4(self.regressor(use_diag=True))

    def bruteforce_inference_one_interaction(self, X, w, order, use_diag):
        n_obj, n_feat = X.shape
        ans = np.zeros(n_obj)
        if order == 2:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    x_prod = X[:, i] * X[:, j]
                    w_prod = np.sum(w[1][i, :] * w[1][j, :])
                    denominator = 2.0**(order-1) if use_diag else 1.0
                    ans += x_prod * w_prod / denominator
        elif order == 3:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    for k in range(0 if use_diag else j+1, n_feat):
                        x_prod = X[:, i] * X[:, j] * X[:, k]
                        w_prod = np.sum(w[2][i, :] * w[2][j, :] * w[2][k, :])
                        denominator = 2.0**(order-1) if use_diag else 1.0
                        ans += x_prod * w_prod / denominator
        elif order == 4:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    for k in range(0 if use_diag else j+1, n_feat):
                        for ell in range(0 if use_diag else k+1, n_feat):
                            x_prod = X[:, i] * X[:, j] * X[:, k] * X[:, ell]
                            w_prod = np.sum(w[3][i, :] * w[3][j, :] * w[3][k, :] * w[3][ell, :])
                            denominator = 2.0**(order-1) if use_diag else 1.0
                            ans += x_prod * w_prod / denominator
        else:
            assert False
        return ans

    def bruteforce_inference(self, X, w, b, use_diag):
        assert len(w) <= 4
        ans = X.dot(w[0]).flatten() + b
        if len(w) > 1:
            ans += self.bruteforce_inference_one_interaction(X, w, 2, use_diag)
        if len(w) > 2:
            ans += self.bruteforce_inference_one_interaction(X, w, 3, use_diag)
        if len(w) > 3:
            ans += self.bruteforce_inference_one_interaction(X, w, 4, use_diag)
        return ans


if __name__ == '__main__':
    unittest.main()
