.. _model-config:

Model Configuration Parameters
==============================

In addition to the feature-specific configurations, there are additional parameters set at the model level.  These are similar to those available in other boosting packages.

* **mode**: This can be 'classification' or 'regression'.  StructureBoost uses a single class to handle both situations.

* **num_trees**: The number of trees to build.  We recommend using an eval_set with early stopping so that the number of trees built is learned dynamically.  When using that option, you can set num_trees to a large value.  However, one can also specify a set number of trees.

* **max_depth**: The maximum depth to build the trees.  The larger the number, the more likely you are to overfit, and the less likely to underfit.  Default is 3.

* **learning_rate**: The "step size" to use when adding each tree.  We recommend erring on the side of having smaller steps and more trees (and using early stopping with an eval_set to optimize model size).  StructureBoost works particularly well under these conditions since it will have multiple opportunities to visit the space of possible splits.

* **subsample**: How much of the training data to use at each tree.  Will be interpreted as a number of rows if given an integer >1 and a percentage of the data if given a float between 0 and 1.  Rows can be chosen with or without replacement, depending on the value of `replace`.

* **replace**: If True, the data set for each tree will be chosen with replacement (as in the bootstrap).  If False, the rows for each tree will chosen without replacement.

* **loss_fn**: By default, we will use log loss (aka entropy, categorical cross-entropy, maximum likelihood) for classification and mean squared error for regression.  To specify a custom loss_fn, one can pass a tuple containing two (vectorized) functions that return the first and second derivatives of the loss_fn.

* **feat_sample_by_tree**: The fraction (or number) of features to sample from at each tree.  Will be interpreted as a number of features for integers>1 and a fraction for floats between 0 and 1.

* **feat_sample_by_node**: The fraction (or number) of features to sample from at each node.  Will be interpreted as a number of features for integers>1 and a fraction for floats between 0 and 1. Effect is cumulative with feat_sample_by_tree - chooses a subset relative to the size of the subset passed to it for each tree.

* **gamma**: Regularization parameter as in XGBoost.  If the best split results in a gain less than gamma, it will not be executed.

* **reg_lambda**: L2 Regularization parameter as in XGBoost.  Serves to "shrink" the size of the value at each leaf.

* **na_unseen_action**: How to choose which way Missing Values should be sent in the tree, if there are no Missing Values at that node at the time of choosing a split.  Default is "weighted_random", which randomly fixes the direction in proportion to the number of data points that went into each direction.  Alternative is "random", which chooses a direction with equal probability regardless of how many data points went each direction.

* **min_sample_split**: How many data points a node must have to consider splitting it further.  Default is 2.
