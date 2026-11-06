"""supgrow: a sup-score test for deciding when (and where) a neural network should grow.

Main entry points
-----------------
grow(X, y, ...)              forward growth of a one-hidden-layer tanh network
sup_score_test(X, y, K, ...) one test of H0: K hidden units are sufficient

Modules
-------
supgrow.growth_test   batched fitting, the sup-score statistic and competing rules
supgrow.deep_growth   layer-wise growth of deep networks (vector test)
supgrow.cnn_growth    exploratory growth of a convolutional network
supgrow.experiments   the experiments of the paper (command line)
"""
__version__ = "1.1.0"

from supgrow.api import grow, sup_score_test, GrowthResult, default_scale

__all__ = ["grow", "sup_score_test", "GrowthResult", "default_scale", "__version__"]
