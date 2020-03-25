StructureBoost FAQ
==================

#. **What is the** `feature_configs`?: StructureBoost uses a configuration dictionary to specify many parameters.  For each feature, we must configure how to determine which splits to evaluate.  For categorical variables, this can particularly detailed.  An advantage of this is that StructureBoost allows a level of fine-tuned control not typically available in other packages.  To reduce the overhead involved, StructureBoost provides tools for quickly defining an "start" configuration that can be easily modified.  Furthermore, we can validate the configuration to reduce any unexpected errors or results.  For more detailed documentation on the various settings visit `Feature Configuration Documentation`_.  Or better yet, go through this `example notebook`_.

.. _example notebook: http://github.com/numeristical/structureboost/examples
.. _Feature Configuration Documentation: http://github.com/numeristical/structureboost
