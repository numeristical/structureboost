import graphs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_basic_config_series(feature_vec, default_config):
    config_dict = {}
    graph_out = None
    if feature_vec.dtype == 'O':
        config_dict['feature_type'] = 'categorical_str'
        config_dict['graph'] = graphs.complete_graph(pd.unique(
                                                     feature_vec.dropna()))
        categorical_method = default_config['default_categorical_method']
        config_dict['split_method'] = categorical_method
        if categorical_method == 'contraction':
            config_dict['contraction_size'] = default_config[
                                                'default_contraction_size']
            config_dict['max_splits_to_search'] = default_config[
                                'default_contraction_max_splits_to_search']
        if categorical_method == 'span_tree':
            config_dict['num_span_trees'] = default_config[
                                                    'default_num_span_trees']
    else:
        config_dict['feature_type'] = 'numerical'
        config_dict['max_splits_to_search'] = default_config[
                                    'default_numerical_max_splits_to_search']
    return config_dict, graph_out


def get_basic_config(X_tr, default_config, X_te=None):
    """Returns a feature_configs based on a training set and some defaults.

    This is a tool to avoid constructing the feature_configs from scratch.
    Call `get_basic_config` with the results of `default_config_dict()`
    as the second argument.
    Then modify the resulting config to your liking.

    Parameters
    ----------

    X_tr : DataFrame
        A dataframe containing the features you plan to train on.  The function
        will analyze the values, make some assumptions, and apply the defaults
        to give a starting configuration dict, which can either be used directly
        of further modified.

    default_config : dict
        This should usually be the output of the `default_config_dict` (or a 
        modified version of it).

    Examples
    --------
    >>> def_set = stb.default_config_dict()
    >>> def_set
    {'default_categorical_method': 'span_tree',
    'default_num_span_trees': 1,
    'default_contraction_size': 9,
    'default_contraction_max_splits_to_search': 25,
    'default_numerical_max_splits_to_search': 25}
    >>> feat_cfg = stb.get_basic_config(X_train, def_set)
    >>> feat_cfg
    {'county': {'feature_type': 'categorical_str',
      'graph': <graphs.graph_undirected at 0x10ea75860>,
      'split_method': 'span_tree',
      'num_span_trees': 1},
     'month': {'feature_type': 'numerical', 'max_splits_to_search': 25}}
    >>> stb_model = stb.StructureBoost(num_trees = 2500,
                                    learning_rate=.02,
                                    feature_configs=feat_cfg, 
                                    max_depth=2,
                                    mode='classification')
    >>> stb_model.fit(X_train, y_train)
"""
    feature_config_dict = {}
    for colname in X_tr.columns:
        if X_te is not None:
            vec_to_use = pd.concat((X_tr[colname], X_te.colname))
        else:
            vec_to_use = X_tr[colname]
        config, graph = get_basic_config_series(vec_to_use, default_config)
        feature_config_dict[colname] = config
        if graph is not None:
            feature_config_dict[colname]['graph'] = graph
    return feature_config_dict


def default_config_dict():
    """Returns a dict of defaults to be used with `get_basic_config`

    The dictionary returned will contain a set of default values.
    These can be modified before the dictionary is used with the 
    `get_basic_config` function.

    Returns
    -------

    config_dict : dict
        A dictionary containing defaults to be used in `get_basic_config()`

    Examples
    --------
    >>> def_set = stb.default_config_dict()
    >>> def_set
    {'default_categorical_method': 'span_tree',
    'default_num_span_trees': 1,
    'default_contraction_size': 9,
    'default_contraction_max_splits_to_search': 25,
    'default_numerical_max_splits_to_search': 25}
    >>> feat_cfg = stb.get_basic_config(X_train, def_set)
    >>> feat_cfg
    {'county': {'feature_type': 'categorical_str',
      'graph': <graphs.graph_undirected at 0x10ea75860>,
      'split_method': 'span_tree',
      'num_span_trees': 1},
     'month': {'feature_type': 'numerical', 'max_splits_to_search': 25}}
    >>> stb_model = stb.StructureBoost(num_trees = 2500,
                                    learning_rate=.02,
                                    feature_configs=feat_cfg, 
                                    max_depth=2,
                                    mode='classification')
    >>> stb_model.fit(X_train, y_train)
    """
    config_dict = {}
    config_dict['default_categorical_method'] = 'span_tree'
    config_dict['default_num_span_trees'] = 1
    config_dict['default_contraction_size'] = 9
    config_dict['default_contraction_max_splits_to_search'] = 25
    config_dict['default_numerical_max_splits_to_search'] = 25
    return config_dict


def apply_defaults(config_dict, default_conf):
    for feature in config_dict.keys():
        config_dict[feature] = apply_defaults_featurewise(
                                config_dict[feature], default_conf)
    return config_dict


def apply_defaults_featurewise(feature_config_dict, default_conf):
    if feature_config_dict['feature_type'] != 'numerical':
        if 'split_method' not in feature_config_dict.keys():
            feature_config_dict['split_method'] = default_conf[
                                            'default_categorical_method']
        method = feature_config_dict['split_method']
        if method == 'span_tree':
            if 'num_span_tree' not in feature_config_dict.keys():
                feature_config_dict['num_span_trees'] = default_conf[
                                            'default_num_span_trees']
        if method == 'contraction':
            if 'contraction_size' not in feature_config_dict.keys():
                feature_config_dict['contraction_size'] = default_conf[
                                            'default_contraction_size']
            if 'max_splits_to_search' not in feature_config_dict.keys():
                feature_config_dict['max_splits_to_search'] = default_conf[
                                    'default_categorical_max_splits_to_search']
    else:
        if 'max_splits_to_search' not in feature_config_dict.keys():
            feature_config_dict['max_splits_to_search'] = default_conf[
                                    'default_numerical_max_splits_to_search']
    return(feature_config_dict)


def ice_plot(model, data_row, column, range_pts, add_nan=True):
    pred_df = pd.DataFrame(columns=data_row.index)
    pred_df.loc[0] = data_row.copy()
    pred_df = pd.concat([pred_df]*len(range_pts), ignore_index=True)
    pred_df[column] = range_pts
    if add_nan:
        new_df = pd.DataFrame(columns=data_row.index)
        new_dr = data_row.copy()
        new_dr[column] = np.nan
        new_df.loc[0] = new_dr
        pred_df = pd.concat((pred_df, new_df))
    pred_vals = model.predict_proba(pred_df)[:, 1]
    p = plt.plot(range_pts, pred_vals[:len(range_pts)])
    if add_nan:
        plt.hlines(pred_vals[-1],
                   xmin=range_pts[0],
                   xmax=range_pts[int(np.floor(len(range_pts)/4+1))],
                   linestyle='dotted', color=p[0].get_color())

def log_loss(truth, preds, eps=1e-15, rtol=1e-05, atol=1e-08):
    """A log_loss that does not complain if some classes are missing.
    
    Parameters
    ----------

    truth : numpy array or Series
        The true values of the observations

    preds : numpy array
        Generally a 2d array, but a 1d array for binary problems is
        permissible and will be interpreted as the probabilities that
        the case in question is 1 (positive).

    eps : float, default is 1e-15
        The preds will be "clipped" to eps, 1-eps to avoid issues with
        taking the logarithm of zero.

    rtol, atol : floats
        These are passed to the numpy.allclose function for checking
        that the rows of `preds` sum to one.

    """
    if (type(preds)==pd.Series):
        preds = preds.to_numpy()
    if (type(truth)==pd.Series):
        truth = truth.to_numpy()
    if (preds.ndim == 2) and (preds.shape[1]==1):
        preds = preds.reshape(-1)
    if (truth.ndim == 2) and (truth.shape[1]==1):
        truth = truth.reshape(-1)
    preds = np.clip(preds, eps, 1-eps)
    if (preds.ndim==1):
        return (-np.mean(truth*np.log(preds) + (1-truth)*np.log(1-preds)))
    elif (preds.ndim==2):
        row_sums = np.sum(preds, axis=1)
        close_to_one = np.allclose(row_sums, np.ones(len(row_sums)), rtol=rtol, atol=atol)
        if not close_to_one:
            w_str = "rows do not sum to one within specified tolerance (rtol, atol)"
            warnings.warn(w_str)
        truth = truth.astype(np.int64)
        highest_val = np.max(truth)
        if (highest_val > preds.shape[1]):
            w_str = 'Value of {} found in truth, but preds has only {} columns'.format(
                                        highest_val, preds.shape[1])
            warnings.warn(w_str)
            return None
        else:
            prob_of_pred_vec = np.array([preds[i,truth[i]] for i in range(preds.shape[0])])
            return(-np.mean(np.log(prob_of_pred_vec)))
    else:
        w_str = "preds has more than 2 dimensions"
        warnings.warn(w_str)
        return None

