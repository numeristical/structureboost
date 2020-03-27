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
