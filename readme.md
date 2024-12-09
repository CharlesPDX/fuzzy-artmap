# Overview
This repository contains two implementations of the Fuzzy ARTMAP algorithm, both based on Carpenter et al., 1992. The `FuzzyArtMap` implementation in the `fuzzy_artmap` module, is scikit-learn interface compatible and based on vectorized `pytorch` operations for the underlying computation. The `FuzzyArtMap` implementation in the `procedural_fuzzy_artmap` module is a bespoke interface implementation that uses `numpy` for the underlying computation in a linear fashion. See the Implementation Notes for more details.

The Fuzzy ARTMAP algorithm is an interpretable neural network classification algorithm that can be run in an online training mode, where the model can receive training examples at any point - even between inference (prediction) runs. The algorithm can also operate in a classic offline, batch-based training mode, like a typical classifier. Fuzzy ARTMAP returns a prediction of the class/category, without a confidence or weight prediction. In the `fuzzy_artmap.FuzzyArtMap` implementation, the `predict_with_membership` function returns the predicted class/category and a float on the interval [0,1] indicating the Fuzzy Set Membership of the input to the predicted class; while not the same as a confidence prediction it is an indicator of how closely the input matches the predicted category.

The model learned by Fuzzy ARTMAP can be interpreted geometrically, with the learned categories as hyper-rectangles. Depending on the nature of the input, the model can also be interpreted as a series of fuzzy If-Then rules. See the `/examples/step-by-step.ipynb` to understand the geometric interpretation and how the model learns and predicts.

# Usage Notes
All input to the `FuzzyArtMap` classifiers must be on the interval [0,1] and complement encoded - including the labels/categories. `NaN`s and missing values are not tolerated in the input or labels. In the `fuzzy_artmap.FuzzyArtMap` implementation, there are options to automatically scale (`auto_scale`) and automatically complement encode the data (`auto_complement_encode`), these are **not** recommended as they mutate the inputs, and are primarily for the `scikit-learn` compatibility test suite.

The main parameters are `baseline_vigilance` (rho a bar in the literature) and the `learning_rate` (beta in the literature). The `baseline_vigilance` controls how good a match the input must be to a learned category in order to trigger that category, the closer to 1 the better the fit must be. For the `learning_rate`, a value closer to 1 causes fast learning and the category to be aggressively updated, a `learning_rate` of 0 causes no learning to occur. Once a category is learned, the `committed_node_learning_rate` takes over, and controls how fast a learned category should change. The `committed_node_learning_rate` can also be set to 1, to enable fast learning. By default it is `0.75` to enable fast learning, slow recode (see Carpenter et al., 1992).

Examples of usage can be seen in the tests and the `circle_test.ipynb`, `circle_square_test.py`, `spiral_test.ipynb` files.

# Implementation Notes
Both implimentations of the Fuzzy ARTMAP algorithm are based on 
[Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
"Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"](https://open.bu.edu/bitstream/handle/2144/2071/91.016.pdf)
IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713. The procedural implementation (`procedural_fuzzy_artmap.py`) is specifically based on [fuzzyartmap_demo.m](https://www2.bcs.rochester.edu/sites/raizada/Matlab/Neural_nets/fuzzyartmap_demo.m) from [here](https://www2.bcs.rochester.edu/sites/raizada/matlab-neural-nets.html) by [Dr. Rajeev Raizada](https://rajeevraizada.github.io/index.html), with some modifications for registering the committed nodes, changing the row/column major ordering for `numpy` vs. matlab, and changing to be a little more Python idiomatic.

The `fuzzy_artmap.FuzzyArtMap` implementation also implements `max_nodes` functionality described in
[Carpenter, G.A., Grossberg, S., & Reynolds, J.H. (1995). A fuzzy ARTMAP nonparametric probability estimator for nonstationary pattern recognition problems. IEEE Transactions on Neural Networks, 6, 1330‑1336. Technical Report CAS/CNS‑93‑047, Boston, MA: Boston University.](https://open.bu.edu/bitstream/handle/2144/2024/93.047.pdf). Normally, the number of category (F2) nodes is allowed to grow unbounded; however, this may not be desirable. In Carpenter et al. (1995), the idea of fixing the number of nodes allowed was introduced. The `fuzzy_artmap.FuzzyArtMap` implementation realizes this through the `max_nodes` parameter, allowing the user to fix the maximum size of the F2 field.

# Examples (in `/examples`)
 NB: All notebooks use `matplotlib` for visualization and runnable top to bottom. Notebooks may require running one cell at a time, but still are runnable top to bottom without any special ordering. See the `/examples/requirements.txt` for notebook specific packages.

### step-by-step.ipynb
This notebook takes a step-by-step approach linking the sections from the Carpenter et al. (1992) to manual implementation and visualization of Fuzzy ARTMAP and the learned model, through several iterations, with points that align with the expected category and points that do not align with the expected category.

### abridged-step-by-step.ipynb
This notebook is a shorter version of the `step-by-step.ipynb` that omits the paper screen grabs, and produces the step-by-step manual calculations that are ultimately used in the `test_fuzzy_artmap_matches_expected_learning` test (`/tests/test_fuzzy_artmap.py`).

### circle_test.ipynb and circle_square_test.py
These are recapitulations of the circle test from Carpenter et al. (1992), illustrating how to use the module in local processing mode. The Jupyter notebook represents the learned categories graphically, while the script runs the training/test set steps.

### procedural_circle_square_test.py
This applies the same circle test to a more procedural implementation of Fuzzy ARTMAP, which is based around numpy.

### spiral_test.ipynb

This notebook illustrates the spiral test from Carpenter et al. (1992).

# Tests
### test_interface.py
Tests the `FuzzyArtMap` implementation's [fidelity](https://scikit-learn.org/dev/modules/generated/sklearn.utils.estimator_checks.check_estimator.html) to the scikit-learn interface. Two tests are skipped `check_methods_subset_invariance` and `check_methods_subset_invariance`, because `FuzzyArtMap` requires continuous values on the interval `[0, 1]` labels must be converted from text to numeric ([`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)) and then [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), values must be converted using the `MinMaxScaler` - these enable the rest of the tests to proceed successfully, but do mutate the input data which violates the expectations of these two tests. Currently, this is implemented using `try/except` functionality, as skipping tests is introduced in version 1.6 of scikit-learn.

### test_fuzzy_artmap.py
Test the core implementation of the model, for node selection, prediction, growing F2, max node support (See Carpenter et al., 1995), and expected calculations.

### test_model_persistence.py
Validate saving and loading the model.

### test_peripheral_functions.py
Test non-core functionality like vector validation, range validation, and complement encoding.

### test_performance.py
Performance evaluation of `FuzzyArtMap` in terms of true & false positive rate, using pytorch tensors and numpy nd-arrays as inputs. Also, performance test the baseline procedural implementation of Fuzzy ARTMAP.