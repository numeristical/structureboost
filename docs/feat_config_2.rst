.. _feature-config-2:

Feature Configuration Parameters
================================

This page will walk through some of the parameters associated with each feature.

Feature Type:  The most important aspect of a feature that must be defined is its type.  StructureBoost admits the following values for the *feature_type*:

* **numerical**: This is a standard numeric variable.  Missing values are permitted.
* **categorical_str**: A Categorical variable, where the values are of type `str`.  Missing values are permitted.  This is the more flexible categorical option, though it can be slightly slower than `categorical_int`.  A *graph* must be provided to indicate the structure of the variable.  If the structure is not known, we recommend using a complete graph (all values are pairwise adjacent).
* **categorical_int**: A Categorical variable where the different values are non-negative integers.  Missing values are *not* permitted (primarily due to fact that numpy's nan does not have an integer option).  This will be slightly faster than categorical_int, so in some cases it may be worth re-coding a string valued variable into integers.  A *graph* must be provided to indicate the structure of the variable.  If the structure is not known, we recommend using a complete graph (all values are pairwise adjacent).
* **graphical_voronoi**:  This is a more complicated type which converts multiple numerical variables into a single categorical feature, with the structure determined by the Voronoi graph.  See `this example`_ for details.

Depending on the feature_type, there will be further parameters that must be specified:

* For `numerical` feature type:

   * `max_splits_to_search`: This parameter must be specified to determine the maximum number of splits that should be checked.  This must be set to a positive integer or np.Inf (to search all splits).  Results are generally better when keeping this to a smaller value (between 10 and 50).  In addition to being faster, it serves to regularize the model.  Splits are chosen randomly.

* For `categorical_int` and `categorical_str` feature types:

   * `split_method`: Options are `span_tree`, `contraction`, and `one_hot`.  This determines how to arrive at the space of splits to search for a categorical variable.  We describe them below:

      * `span_tree`:  This is the recommended method.  A graph must be provided.  A number of spanning trees of the graph will be chosen uniformly from the space of all spanning trees.  Then, the algorithm will consider the splits induced by removing each possible edge from the spanning tree.  This method requires the specification of `num_span_trees` - the number of spanning trees to check for this feature at each node.  The recommended value of `num_span_trees` is 1.

      * `contraction`: This method also requires a graph.  The graph is reduced in size by randomly contracting edges until it reaches a size <= the specified parameter `contraction_size`.  At that point, it enumerates all possible binary splits into two connected components of the contracted graph.  A number `max_splits_to_search` of these a randomly selected to be evaluated.  Enumerating all possible (allowable) binary splits is computationally intensive, therefore it is recommended to choose a contraction_size around 9.  Similarly, we recommend setting `max_splits_to_search` in the range of 10 to 50.  If the graph is a cycle or similarly sparse, then larger numbers can be chosen.  To enumerate all allowable binary splits, choose the `contraction_size` to be larger than the size of the graph.  To evaluate all enumerated splits, choose `max_splits_to_search`

      * `one_hot`: This method will search all "one-hot" splits -- that is, splits where one value goes left and the rest go right.  This is not a recommended method, but provided for comparison purposes.  If this method is chosen, a graph is not required to be specified.

.. _this example: https://github.com/numeristical/structureboost/tree/master/examples/MoCap_Spatial.ipynb
