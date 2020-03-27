.. _feature-configuration:

Feature Configuration
=====================

StructureBoost allows some options to be configured at a feature-specific level.  These options are contained in a dictionary object where the keys of the dictionary contain the names of the features for the model.  An example configuration dictionary is below:

.. code-block:: python

   {'county': {'feature_type': 'categorical_str',
    'graph': <graphs.graph_undirected at 0x13cf2d080>,
    'split_method': 'span_tree',
    'num_span_trees': 1},
    'month': {'feature_type': 'numerical', 'max_splits_to_search': 25}}

In this example, there are two features: 'county' and 'month'.  The feature 'county' has feature type 'categorical_str', with 'split_method' set to 'span_tree' and 'num_span_trees' set to 1.  It also has a 'graph' associated with it.  The feature 'month' is of feature_type 'numerical' and has the parameter 'max_splits_to_search' set to 25.
