"""Decision Tree based on Discrete Graph structure"""
import graphs
import copy
import random
import warnings
import numpy as np
import scipy as sp
import pandas as pd
from libc.math cimport log as clog
from libc.math cimport isnan
cimport numpy as cnp
cimport cython


class StructureDecisionTree(object):
    """Decision Tree using graphical structure.

    Uses Newton steps based on first and second derivatives of loss fn.
    """

    def __init__(self, feature_configs, feature_graphs, min_size_split=2,
                 max_depth=3, gamma=0, feat_sample_by_node=1,
                 reg_lambda=1):
        self.dec_tree = {}
        self.feature_configs = feature_configs
        self.dec_tree['feature_graphs'] = feature_graphs
        self.num_leafs = 0
        self.min_size_split = min_size_split
        self.max_depth = max_depth
        self.gamma = gamma
        self.feat_sample_by_node = feat_sample_by_node
        self.reg_lambda = reg_lambda
        self.node_summary_fn = _node_summary_gh
        self.feat_list_full = list(self.feature_configs.keys())

    def fit(self, X_train, g_h_train, feature_sublist, uv_dict):
        # Tree fitting works through a queue of nodes to process
        # called (node_to_proc_list)
        # The initial node is just the root of the tree
        col_list = list(X_train.columns)
        self.train_column_to_int_dict = {col_list[i]: i
                                         for i in range(len(col_list))}
        self.node_to_proc_list = [self.dec_tree]

        # Initialize values to what they are at the root of the tree
        self.dec_tree['depth'] = 0
        self.dec_tree['mask'] = np.ones(g_h_train.shape[0]).astype(bool)
        self.feature_sublist = feature_sublist
        self.X_train = X_train
        self.g_h_train = g_h_train
        self.uv_dict = uv_dict

        # Process nodes until none are left to process
        while self.node_to_proc_list:
            node_to_process = self.node_to_proc_list.pop()
            self._process_tree_node(node_to_process)

    def predict(self, X_test):
        col_list = list(X_test.columns)
        column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        return get_prediction(self.dec_tree, X_test.to_numpy(),
                              column_to_int_dict)

    def _process_tree_node(self, curr_node):
        # Restrict to relevant data for the node in question
        X_train_node = self.X_train[curr_node['mask']]
        g_h_train_node = self.g_h_train[curr_node['mask']]
        g_train_node = g_h_train_node[:, 0]
        h_train_node = g_h_train_node[:, 1]

        # Save information about the current node
        num_dp = len(g_train_node)
        curr_node['num_data_points'] = num_dp
        curr_node['node_summary_val'] = _node_summary_gh(g_train_node,
                                                         h_train_node,
                                                         self.gamma)

        # Check if node is eligible to be split further
        wrap_up_now = ((num_dp < self.min_size_split) or
                       (curr_node['depth'] >= self.max_depth))
        if wrap_up_now:
            self._wrap_up_node(curr_node, g_train_node, h_train_node)
            return None

        features_to_search = self._get_features_to_search()
        best_split_dict = _initialize_best_split_dict()

        # Main loop over features to find best split
        for feature in features_to_search:
            best_split_for_feature = evaluate_feature(
                                        self.feature_configs[feature],
                                        curr_node['feature_graphs'],
                                        feature, X_train_node,
                                        g_train_node, h_train_node,
                                        self.gamma, self.reg_lambda,
                                        self.uv_dict)
            if best_split_for_feature:
                best_split_for_feature['split_feature'] = feature
                if (best_split_for_feature['loss_score'] < best_split_dict[
                                                            'loss_score']):
                    best_split_dict = best_split_for_feature

        # Execute the split (if valid split is found) else wrap-up
        if best_split_dict['loss_score'] < np.inf:
            self._execute_split(curr_node, best_split_dict,
                                curr_node['feature_graphs'])
        else:
            self._wrap_up_node(curr_node, g_train_node, h_train_node)

    def _get_features_to_search(self):
        if self.feat_sample_by_node < 1:
            feat_set_size = self.feat_sample_by_node * len(
                                                self.feature_sublist)
            feat_set_size = int(np.maximum(feat_set_size, 1))
            np.random.shuffle(self.feature_sublist)
            features_to_search = self.feature_sublist[:feat_set_size]
        elif self.feat_sample_by_node > 1:
            feat_set_size = int(self.feat_sample_by_node)
            np.random.shuffle(self.feature_sublist)
            features_to_search = self.feature_sublist[:feat_set_size]
        else:
            features_to_search = self.feature_sublist
        return features_to_search

    def _wrap_up_node(self, curr_node, g_train_node, h_train_node):
        # Compute summary stats of node and mark it as a leaf
        curr_node['node_summary_val'] = _node_summary_gh(g_train_node,
                                                         h_train_node,
                                                         self.reg_lambda)
        curr_node['num_data_points'] = len(g_train_node)
        curr_node['node_type'] = 'leaf'
        self.num_leafs += 1
        curr_node.pop('mask')

    def _execute_split(self, curr_node, best_split_dict, feature_graphs_node):
        ft = best_split_dict['feature_type']
        if ft == 'numerical':
            self._execute_split_numerical(curr_node, best_split_dict,
                                          feature_graphs_node)
        elif (ft == 'categorical_str') or (ft == 'categorical_int'):
            self._execute_split_graphical(curr_node, best_split_dict,
                                          feature_graphs_node)
        elif ft == 'graphical_voronoi':
            self._execute_split_voronoi(curr_node, best_split_dict,
                                        feature_graphs_node)
        else:
            print("Unknown feature type")

    def _execute_split_numerical(self, curr_node, best_split_dict,
                                 feature_graphs_node):
        curr_feat = best_split_dict['split_feature']
        split_val = best_split_dict['left_split']
        if pd.isnull(split_val):
            if best_split_dict['na_left'] == 1:
                left_mask = pd.isnull(self.X_train[curr_feat])
            else:
                left_mask = np.logical_not(np.isnan(self.X_train[curr_feat]))
        else:
            left_mask = (self.X_train[curr_feat] <= split_val).values

        # left_mask = (self.X_train[best_split_dict['split_feature']] <=
        #              best_split_dict['left_split']).values
        curr_node['split_val'] = split_val
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        curr_node['na_left'] = best_split_dict['na_left']
        curr_node['na_dir_random'] = best_split_dict['na_dir_random']
        curr_mask = curr_node.pop('mask')

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_right = feature_graphs_node.copy()

        self._create_children_nodes(curr_node, feature_graphs_left,
                                    feature_graphs_right,
                                    curr_mask, left_mask)

    def _create_children_nodes(self, curr_node, feature_graphs_left,
                               feature_graphs_right,
                               curr_mask, left_mask):
        # Create left and right children
        curr_node['left_child'] = {}
        curr_node['left_child']['depth'] = curr_node['depth'] + 1
        curr_node['left_child']['mask'] = curr_mask & left_mask
        curr_node['left_child']['feature_graphs'] = feature_graphs_left

        curr_node['right_child'] = {}
        curr_node['right_child']['depth'] = curr_node['depth'] + 1
        curr_node['right_child']['mask'] = (curr_mask &
                                            np.logical_not(left_mask))
        curr_node['right_child']['feature_graphs'] = feature_graphs_right

        # Add left and right children to queue
        self.node_to_proc_list.append(curr_node['left_child'])
        self.node_to_proc_list.append(curr_node['right_child'])

    def _execute_split_graphical(self, curr_node, best_split_dict,
                                 feature_graphs_node):

        feat_vec_train = self.X_train[best_split_dict['split_feature']]
        left_mask = feat_vec_train.isin(
                        best_split_dict['left_split']).values
        if ('na_left' in best_split_dict.keys()) and (
                            best_split_dict['na_left'] == 1):
            na_left_override = pd.isnull(feat_vec_train)
            left_mask = left_mask | na_left_override

        # record info about current node
        curr_node['left_split'] = best_split_dict['left_split']
        curr_node['right_split'] = (feature_graphs_node[
                                best_split_dict['split_feature']].vertices -
                                best_split_dict['left_split'])
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        if 'na_left' in best_split_dict.keys():
            curr_node['na_left'] = best_split_dict['na_left']
        else:
            curr_node['na_left'] = int(random.random() < .5)
        if 'na_dir_random' in best_split_dict.keys():
            curr_node['na_dir_random'] = best_split_dict['na_dir_random']
        else:
            curr_node['na_dir_random'] = 1
        curr_mask = curr_node.pop('mask')

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        new_left_graph = (
            graphs.get_induced_subgraph(
                feature_graphs_left[curr_node['split_feature']],
                curr_node['left_split']))
        feature_graphs_left[curr_node['split_feature']] = (
                graphs.get_induced_subgraph(
                        feature_graphs_left[curr_node['split_feature']],
                        curr_node['left_split']))

        feature_graphs_right = feature_graphs_node.copy()
        new_right_graph = (
              graphs.get_induced_subgraph(
                feature_graphs_right[curr_node['split_feature']],
                curr_node['right_split']))
        feature_graphs_right[curr_node['split_feature']] = (
                graphs.get_induced_subgraph(
                        feature_graphs_right[curr_node['split_feature']],
                        curr_node['right_split']))
        self._create_children_nodes(curr_node, feature_graphs_left,
                                    feature_graphs_right, curr_mask,
                                    left_mask)

    def _execute_split_voronoi(self, curr_node, best_split_dict,
                               feature_graphs_node):

        subfeature_indices = [self.train_column_to_int_dict[colname]
                              for colname in best_split_dict[
                                                'subfeature_list']]
        data_array = self.X_train.values[:, subfeature_indices]
        tmp_tr = best_split_dict['voronoi_kdtree'].query(data_array)
        feat_vec = tmp_tr[1].astype(np.int64)
        fs_array = np.fromiter(best_split_dict['left_split'], int,
                               len(best_split_dict['left_split']))
        vec_len = len(feat_vec)
        lsplit_len = len(fs_array)
        left_mask = np.zeros(vec_len, dtype=np.int64)
        left_mask = get_mask_int_c_alt(feat_vec, fs_array, vec_len,
                                       lsplit_len, left_mask)

        # record info about current node
        curr_node['left_split'] = best_split_dict['left_split']
        curr_node['right_split'] = (best_split_dict['voronoi_graph'].vertices -
                                    best_split_dict['left_split'])
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']
        curr_node['subfeature_list'] = best_split_dict['subfeature_list']
        curr_node['voronoi_kdtree'] = best_split_dict['voronoi_kdtree']
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        curr_mask = curr_node.pop('mask')

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_right = feature_graphs_node.copy()

        self._create_children_nodes(curr_node, feature_graphs_left,
                                    feature_graphs_right, curr_mask, left_mask)


def _initialize_best_split_dict():
    out_dict = {}
    out_dict['loss_score'] = np.inf
    out_dict['left_split'] = None
    out_dict['split_feature'] = None
    return(out_dict)


def root_mean_squared_error(vec1, vec2):
    return np.sqrt(np.mean((vec1-vec2)**2))


def _get_gh_score_num(double g_left,  double g_right,
                      double h_left, double h_right,
                      double gamma, double reg_lambda, tol=1e-12):
    loss_val = -1.0 * (.5*(((g_left*g_left)/(h_left+reg_lambda)) +
                       ((g_right*g_right)/(h_right+reg_lambda)) -
                   (((g_left + g_right)*(g_left + g_right)) /
                    (h_left + h_right+reg_lambda)))-gamma)
    if loss_val >= -tol:
        loss_val = np.inf
    return(loss_val)


def _get_gh_score_array(cnp.ndarray[double] g_left,
                        cnp.ndarray[double] g_right,
                        cnp.ndarray[double] h_left,
                        cnp.ndarray[double] h_right,
                        double gamma, double reg_lambda):
    return(.5*(((g_left*g_left)/(h_left+reg_lambda)) +
               ((g_right*g_right)/(h_right+reg_lambda)) -
               (((g_left+g_right) * (g_left+g_right)) /
                (h_left+h_right+reg_lambda)))-gamma)


def _node_summary_gh(y_vec_g, y_vec_h, reg_lambda):
    if (len(y_vec_g) == 0) or (len(y_vec_h) == 0):
        print('warning got 0 length vec in node summary')
        return 0
    else:
        out_val = -((np.sum(y_vec_g))/(np.sum(y_vec_h)+reg_lambda))
        return(out_val)


def evaluate_feature(feature_config, feature_graphs, feature_name,
                     X_train_node, g_train_node, h_train_node,
                     gamma, reg_lambda, uv_dict):

    ft = feature_config['feature_type']

    if ft == 'numerical':
        return _evaluate_feature_numerical(
                        feature_config, X_train_node[feature_name].values,
                        g_train_node, h_train_node, gamma, reg_lambda,
                        uv_dict[feature_name])
    elif ((ft == 'categorical_int') or (ft == 'categorical_str')):
        split_method = feature_config['split_method']
        if split_method == 'contract_enum':
            return _evaluate_feature_enum_contr(
                        feature_config, feature_graphs[feature_name],
                        X_train_node[feature_name].values,
                        g_train_node, h_train_node, gamma, reg_lambda)
        elif split_method == 'span_tree':
            return _evaluate_feature_multitree(
                        feature_config, feature_graphs[feature_name],
                        X_train_node[feature_name].values,
                        g_train_node, h_train_node, gamma, reg_lambda)
        elif split_method == 'onehot':
            return _evaluate_feature_onehot(
                            feature_config, feature_graphs[feature_name],
                            X_train_node[feature_name].values,
                            g_train_node, h_train_node, gamma, reg_lambda)
        else:
            w_str = 'Unknown split method "{}" - ignoring feature'.format(
                                                                split_method)
            warnings.warn(w_str)
    elif ft == 'graphical_voronoi':
        return _evaluate_feature_voronoi(
                        feature_config, X_train_node,
                        g_train_node, h_train_node, gamma, reg_lambda)
    else:
        warnings.warn('Unknown feature type "{}": ignoring'.format(ft))


def _evaluate_feature_numerical(feature_config, feature_vec,
                                g_vec, h_vec, gamma, reg_lambda, uv_array):
    interp_mode = ('random' if 'interp_mode' not in feature_config.keys()
                   else feature_config['interp_mode'])
    splits_to_eval = _get_numerical_splits(feature_vec, uv_array,
                                           interp_mode=interp_mode)
    if len(splits_to_eval) == 0:
        return(_initialize_best_split_dict())
    else:
        splits_to_eval = _subset_splits(splits_to_eval, feature_config)

        best_loss, best_spl_val, na_left, na_rnd = _evaluate_numerical_splits(
                                        feature_vec, g_vec, h_vec,
                                        splits_to_eval, gamma, reg_lambda)
        best_split_of_feat = {}
        best_split_of_feat['loss_score'] = best_loss
        best_split_of_feat['left_split'] = best_spl_val
        best_split_of_feat['feature_type'] = 'numerical'
        best_split_of_feat['na_left'] = na_left
        best_split_of_feat['na_dir_random'] = na_rnd
        return(best_split_of_feat)


def _subset_splits(splits_to_eval, feature_config):
    split_res = (feature_config['max_splits_to_search']
                 if 'max_splits_to_search' in feature_config.keys()
                    else np.Inf)
    split_count = len(splits_to_eval)
    if split_res < split_count:
        split_indices = np.random.choice(split_count, size=split_res)
        sorted_split_indices = np.sort(split_indices)
        # Make sure we include the largest val for NA handling
        if sorted_split_indices[-1] != split_count-1:
            sorted_split_indices = np.concatenate((sorted_split_indices,
                                                  [split_count-1]))
        splits_to_eval = splits_to_eval[sorted_split_indices]
    return splits_to_eval


def _get_numerical_splits(feature_vec, uv_array, interp_mode='random',
                          prec_digits=16):
    # unique_vals = np.sort(pd.unique(feature_vec))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_val, max_val, has_na = max_min_hasnan(feature_vec.astype(float),
                                                  len(feature_vec))
        # min_val = np.nanmin(feature_vec)
        # max_val = np.nanmax(feature_vec)
        # has_na = np.any(np.isnan(feature_vec))
        unique_vals = uv_array[(uv_array >= min_val) & (
                                uv_array <= max_val)]
        if has_na and not np.isnan(max_val):
            unique_vals = np.concatenate((unique_vals, [np.nan]))
    if len(unique_vals) <= 1:
        return []
    else:
        weight = .5 if interp_mode == 'mid' else random.random()
        splits = (weight * unique_vals[1:]) + ((1-weight) * unique_vals[:-1])
        return splits


def _evaluate_numerical_splits(feature_vec, g_vec, h_vec,
                               split_vec, gamma, reg_lambda):

    has_na_vals = np.isnan(split_vec[-1])
    bin_result_vec = np.searchsorted(split_vec, feature_vec, side='right')
    g_sum_bins, h_sum_bins = get_bin_sums_c(g_vec, h_vec,
                                            bin_result_vec,
                                            len(split_vec)+1)
    g_sum_total, g_sum_left, g_sum_right = get_left_right_sums(g_sum_bins)
    h_sum_total, h_sum_left, h_sum_right = get_left_right_sums(h_sum_bins)
    score_vec = (-1)*_get_gh_score_array(g_sum_left, g_sum_right,
                                         h_sum_left, h_sum_right,
                                         gamma, reg_lambda)

    best_loss, best_split_val = _get_best_vals(score_vec, split_vec)
    if has_na_vals and (len(split_vec) > 2):
        g_sum_left_nal = (g_sum_left + (g_sum_total - g_sum_left[-1]))[:-1]
        h_sum_left_nal = (h_sum_left + (h_sum_total - h_sum_left[-1]))[:-1]
        g_sum_right_nal = g_sum_total - g_sum_left_nal
        h_sum_right_nal = h_sum_total - h_sum_left_nal

        score_vec_nal = (-1)*_get_gh_score_array(g_sum_left_nal,
                                                 g_sum_right_nal,
                                                 h_sum_left_nal,
                                                 h_sum_right_nal,
                                                 gamma, reg_lambda)
        best_loss_nal, best_split_val_nal = _get_best_vals(score_vec_nal,
                                                           split_vec)
        if best_loss_nal < best_loss:
            # NAs go left by design
            return best_loss_nal, best_split_val_nal, 1, 0
        else:
            # NAs go right by design
            return best_loss, best_split_val, 0, 0
    else:
        # If no NAs at training, randomly choose side for NAs at this node
        # adjust here if we want coin flip to be other than .5
        return best_loss, best_split_val, int(random.random() < .5), 1


def _get_best_vals(score_vec, split_vec):
    best_split_index = np.argmin(score_vec)
    # best_loss = np.min(score_vec)
    best_loss = score_vec[best_split_index]
    best_split_val = split_vec[best_split_index]
    return best_loss, best_split_val


def get_bin_sums_c(cnp.ndarray[double] g_vec, cnp.ndarray[double] h_vec,
                   cnp.ndarray[long] bin_result_vec, long out_vec_size):
    cdef int i
    cdef int m = bin_result_vec.shape[0]

    cdef cnp.ndarray[double] g_sum_bins = np.zeros(out_vec_size)
    cdef cnp.ndarray[double] h_sum_bins = np.zeros(out_vec_size)

    for i in range(m):
        g_sum_bins[bin_result_vec[i]] += g_vec[i]
        h_sum_bins[bin_result_vec[i]] += h_vec[i]
    return g_sum_bins, h_sum_bins


def get_left_right_sums(bin_sums):
    sum_left = np.cumsum(bin_sums)
    sum_total = sum_left[-1]
    sum_left = sum_left[:-1]
    sum_right = sum_total - sum_left
    return sum_total, sum_left, sum_right


def _evaluate_feature_voronoi(feature_config, X_train_node, g_train_node,
                              h_train_node, gamma, reg_lambda):

    # pull and process config items
    feature_type = feature_config['feature_type']
    sub_features = feature_config['subfeature_list']
    vor_sample_size = feature_config['voronoi_sample_size']
    vor_min_sample_size = feature_config['voronoi_min_pts']
    if (X_train_node.shape[0] < vor_min_sample_size):
        return(_initialize_best_split_dict())
    vor_sample_size = np.minimum(vor_sample_size, X_train_node.shape[0])

    # Get feature graph, kd-tree, and subset of columns
    feature_graph, vor_kdtree, data_array = _get_graph_kd_tree(X_train_node,
                                                               sub_features,
                                                               vor_sample_size)
    # Map data points to regions
    tmp_tr = vor_kdtree.query(data_array)
    feature_vec_node = tmp_tr[1].astype(np.int64)

    if feature_config['split_method'] == 'span_tree':
        best_split_of_feat = _evaluate_feature_multitree(feature_config,
                                                         feature_graph,
                                                         feature_vec_node,
                                                         g_train_node,
                                                         h_train_node,
                                                         gamma, reg_lambda)
    elif feature_config['split_method'] == 'contract_enum':
        best_split_of_feat = _evaluate_feature_enum_contr(feature_config,
                                                          feature_graph,
                                                          feature_vec_node,
                                                          g_train_node,
                                                          h_train_node,
                                                          gamma, reg_lambda)
    else:
        print('Unknown method for splitting feature')

    best_split_of_feat['voronoi_kdtree'] = vor_kdtree
    best_split_of_feat['voronoi_graph'] = feature_graph
    best_split_of_feat['subfeature_list'] = sub_features
    return(best_split_of_feat)


def _get_graph_kd_tree(X_train_node, sub_features, vor_sample_size):
    data_array = X_train_node.loc[:, sub_features].values
    cm = get_corner_mat(data_array)
    voronoi_sample_mat = X_train_node.loc[:, sub_features].sample(
                                vor_sample_size, replace=True).values
    voronoi_sample_mat = np.concatenate((voronoi_sample_mat, cm), axis=0)

    vor_obj = sp.spatial.Voronoi(voronoi_sample_mat)
    voronoi_kdtree = sp.spatial.cKDTree(voronoi_sample_mat)
    feature_graph = graphs.graph_undirected(
                        ridge_points_to_edge_set(vor_obj.ridge_points))
    return feature_graph, voronoi_kdtree, data_array


def _evaluate_feature_multitree(feature_config, feature_graph,
                                feature_vec_node, g_train_node,
                                h_train_node, gamma, reg_lambda):

    if not feature_graph.vertices:
        return(_initialize_best_split_dict())

    g_sum, h_sum, g_val_arr, h_val_arr, map_dict = _init_for_span_trees(
                                                feature_vec_node,
                                                g_train_node, h_train_node,
                                                feature_graph,
                                                feature_config['feature_type'])

    best_split_of_feat = _eval_multiple_span_trees(g_sum, h_sum,
                                                   g_val_arr, h_val_arr,
                                                   map_dict,
                                                   feature_config[
                                                    'num_span_trees'],
                                                   feature_graph,
                                                   feature_config[
                                                    'feature_type'],
                                                   gamma, reg_lambda)
    return(best_split_of_feat)


def _init_for_span_trees(feature_vec_node, g_train_node, h_train_node,
                         feature_graph, feature_type):
    g_sum = np.sum(g_train_node)
    h_sum = np.sum(h_train_node)
    is_integer_valued = (feature_type == 'categorical_int') or (
                         feature_type == 'graphical_voronoi')
    if is_integer_valued:
        max_num_vertices = np.max(np.array(list(feature_graph.vertices)))+1
        map_dict = None
    else:
        max_num_vertices = len(feature_graph.vertices)+1
        sorted_vertices = np.sort(list(feature_graph.vertices))
        map_dict = {sorted_vertices[i]: i for i in range(max_num_vertices-1)}
        map_dict[np.nan] = max_num_vertices-1

    g_val_arr = np.zeros(max_num_vertices)
    h_val_arr = np.zeros(max_num_vertices)
    if is_integer_valued:
        g_val_arr, h_val_arr = get_g_h_feature_sum_arrays(
                                        feature_vec_node,
                                        g_train_node, h_train_node,
                                        g_val_arr, h_val_arr)
    else:
        g_val_arr, h_val_arr = get_g_h_feature_sum_arrays_gen(
                                        feature_vec_node, map_dict,
                                        g_train_node, h_train_node,
                                        g_val_arr, h_val_arr)
    return g_sum, h_sum, g_val_arr, h_val_arr, map_dict


def _eval_multiple_span_trees(g_sum, h_sum, g_val_arr, h_val_arr, map_dict,
                              num_spanning_trees, feature_graph, feature_type,
                              gamma, reg_lambda):
    best_split_of_feat = _initialize_best_split_dict()
    for i in range(num_spanning_trees):
        g_accum_array = g_val_arr.copy()
        h_accum_array = h_val_arr.copy()
        curr_span_tree, drd = feature_graph.get_uniform_random_spanning_tree()
        root_dist_list = [(a, b) for a, b in drd.items()]
        vertex_arr = np.array([x[0] for x in root_dist_list])
        dist_arr = np.array([x[1] for x in root_dist_list])
        vertex_order = vertex_arr[np.argsort(-dist_arr)]
        vertex_to_split_dict = {vertex: set([vertex])
                                for vertex in curr_span_tree.vertices}
        best_split_of_feat = _eval_span_tree(curr_span_tree, g_accum_array,
                                             h_accum_array, g_sum, h_sum,
                                             best_split_of_feat,
                                             vertex_to_split_dict, gamma,
                                             reg_lambda, feature_type,
                                             vertex_order, map_dict)
    return(best_split_of_feat)


def _eval_span_tree(span_tree, g_accum_array, h_accum_array,
                    g_sum, h_sum,
                    best_split_of_feat, vertex_to_split_dict,
                    gamma, reg_lambda, out_feature_type,
                    vertex_order, map_dict=None):
    if map_dict is not None:
        na_pres = ((g_accum_array[map_dict[np.nan]] != 0) or (
                    h_accum_array[map_dict[np.nan]] != 0))
    else:
        na_pres = False

    # First score na vs not_na and compare
    if na_pres:
        g_na = g_accum_array[map_dict[np.nan]]
        h_na = h_accum_array[map_dict[np.nan]]
        na_vs_loss = _get_gh_score_num(g_na, g_sum-g_na,
                                       h_na, h_sum-h_na,
                                       gamma, reg_lambda)
        if na_vs_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = na_vs_loss
            best_split_of_feat['left_split'] = set()
            best_split_of_feat['feature_type'] = out_feature_type
            best_split_of_feat['na_left'] = 1
            best_split_of_feat['na_dir_random'] = 0

    # Go through the order of the tree comparing toward root vs away root
    for leaf_vertex_raw in vertex_order[:-1]:
        leaf_neighbor_raw = next(iter(span_tree.adjacent_vertices(
                                                    leaf_vertex_raw)))
        if map_dict is not None:
            leaf_vertex_ind = map_dict[leaf_vertex_raw]
            leaf_neighbor_ind = map_dict[leaf_neighbor_raw]
        else:
            leaf_vertex_ind = leaf_vertex_raw
            leaf_neighbor_ind = leaf_neighbor_raw

        g_left = g_accum_array[leaf_vertex_ind]
        h_left = h_accum_array[leaf_vertex_ind]
        curr_loss = _get_gh_score_num(g_left, g_sum-g_left,
                                      h_left, h_sum-h_left,
                                      gamma, reg_lambda)
        if na_pres:
            curr_loss_nal = _get_gh_score_num(g_left+g_na,
                                              g_sum-g_left-g_na,
                                              h_left+h_na,
                                              h_sum-h_left-h_na,
                                              gamma, reg_lambda)
            na_left, na_rnd = 0, 0
        else:
            # Could adjust here to make coin flip weighted (on number of obs)
            left_prob = .5
            na_left, na_rnd = int(random.random() < left_prob), 1
            curr_loss_nal = np.Inf
        if curr_loss_nal < curr_loss:
            curr_loss, na_left, na_rnd = curr_loss_nal, 1, 0

        if curr_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = curr_loss
            best_split_of_feat['left_split'] = vertex_to_split_dict[
                                                    leaf_vertex_raw]
            best_split_of_feat['feature_type'] = out_feature_type
            best_split_of_feat['na_left'] = na_left
            best_split_of_feat['na_dir_random'] = na_rnd

        g_accum_array[leaf_neighbor_ind] = g_accum_array[
                                        leaf_neighbor_ind] + g_accum_array[
                                                            leaf_vertex_ind]
        h_accum_array[leaf_neighbor_ind] = h_accum_array[
                                        leaf_neighbor_ind] + h_accum_array[
                                                            leaf_vertex_ind]
        vertex_to_split_dict[leaf_neighbor_raw] = vertex_to_split_dict[
                                                leaf_neighbor_raw].union(
                                                vertex_to_split_dict[
                                                 leaf_vertex_raw])
        new_vertices = span_tree.vertices.copy()
        new_vertices.remove(leaf_vertex_raw)
        new_edges = span_tree.edges.copy()
        new_edges.remove(frozenset([leaf_vertex_raw, leaf_neighbor_raw]))
        span_tree = graphs.graph_undirected(new_edges, new_vertices)
    return(best_split_of_feat)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_g_h_feature_sum_arrays(cnp.ndarray[long] feature_vec_node,
                               cnp.ndarray[double] g_train_node,
                               cnp.ndarray[double] h_train_node,
                               cnp.ndarray[double] g_val_arr,
                               cnp.ndarray[double] h_val_arr):
    cdef long i, ind
    cdef double g_value, h_value
    cdef long array_size = len(feature_vec_node)
    for i in range(array_size):
        ind = feature_vec_node[i]
        g_val_arr[ind] = g_val_arr[ind]+g_train_node[i]
        h_val_arr[ind] = h_val_arr[ind]+h_train_node[i]
    return g_val_arr, h_val_arr


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_g_h_feature_sum_arrays_gen(feature_vec_node, dict mapping_dict,
                                   cnp.ndarray[double] g_train_node,
                                   cnp.ndarray[double] h_train_node,
                                   cnp.ndarray[double] g_val_arr,
                                   cnp.ndarray[double] h_val_arr):
    cdef long i, ind
    cdef double g_value, h_value
    cdef long array_size = len(feature_vec_node)
    for i in range(array_size):
        ind = mapping_dict[feature_vec_node[i]]
        g_val_arr[ind] = g_val_arr[ind]+g_train_node[i]
        h_val_arr[ind] = h_val_arr[ind]+h_train_node[i]
    return g_val_arr, h_val_arr


def _evaluate_feature_enum_contr(feature_config, feature_graph,
                                 feature_vec_node, g_train_node,
                                 h_train_node, gamma, reg_lambda):

    possible_splits = _get_possible_splits(feature_graph,
                                           feature_config['feature_type'],
                                           feature_config['contraction_size'])

    splits_to_eval = _choose_split_subset(possible_splits,
                                          feature_config[
                                                 'max_splits_to_search'])

    best_split_of_feat = _eval_split_set(feature_vec_node, g_train_node,
                                         h_train_node, splits_to_eval,
                                         feature_config['feature_type'],
                                         gamma, reg_lambda)
    return(best_split_of_feat)


def _eval_split_set(feature_vec_node, g_train_node, h_train_node,
                    splits_to_eval, feature_type, gamma, reg_lambda):

    best_split_of_feat = _initialize_best_split_dict()
    na_mask = pd.isnull(feature_vec_node)
    na_pres = np.any(na_mask)

    g_sum = np.sum(g_train_node)
    h_sum = np.sum(h_train_node)
    is_integer_valued = (feature_type == 'categorical_int') or (
                         feature_type == 'graphical_voronoi')
    # Check na vs not_na split
    if na_pres:
        na_rnd = 0
        g_na_masked = g_train_node[na_mask]
        h_na_masked = h_train_node[na_mask]
        curr_loss = _score_split(g_na_masked, h_na_masked, g_sum, h_sum,
                                 gamma, reg_lambda)

        if curr_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = curr_loss
            best_split_of_feat['left_split'] = set()
            best_split_of_feat['feature_type'] = feature_type
            best_split_of_feat['na_left'] = 1
            best_split_of_feat['na_dir_random'] = 0

    # Loop within values of each feature
    for partition in splits_to_eval:
        curr_partition = list(partition)
        left_split = curr_partition[0]
        right_split = curr_partition[1]
        if is_integer_valued:
            fs_array = np.fromiter(left_split, int, len(left_split))
            vec_len = len(feature_vec_node)
            lsplit_len = len(fs_array)
            mask_left = np.zeros(vec_len, dtype=np.int64)
            mask_left = get_mask_int_c_alt(feature_vec_node.astype(np.int64),
                                           fs_array, vec_len, lsplit_len,
                                           mask_left)
        else:
            mask_left = get_mask(feature_vec_node, left_split)
        g_masked = g_train_node[mask_left]
        h_masked = h_train_node[mask_left]
        curr_loss = _score_split(g_masked, h_masked, g_sum, h_sum,
                                 gamma, reg_lambda)

        if curr_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = curr_loss
            best_split_of_feat['left_split'] = left_split
            best_split_of_feat['feature_type'] = feature_type
            if na_pres:
                best_split_of_feat['na_left'] = 0
                best_split_of_feat['na_dir_random'] = 1
            else:
                best_split_of_feat['na_left'] = int(random.random() < .5)
                best_split_of_feat['na_dir_random'] = 1

        if na_pres:
            g_masked_na = g_train_node[mask_left | na_mask]
            h_masked_na = h_train_node[mask_left | na_mask]
            curr_loss = _score_split(g_masked_na, h_masked_na, g_sum, h_sum,
                                     gamma, reg_lambda)
            if curr_loss < best_split_of_feat['loss_score']:
                best_split_of_feat['loss_score'] = curr_loss
                best_split_of_feat['left_split'] = left_split
                best_split_of_feat['feature_type'] = feature_type
                best_split_of_feat['na_left'] = 1
                best_split_of_feat['na_dir_random'] = 0

    return(best_split_of_feat)


def _choose_split_subset(possible_splits, msts, replace=True):
    nps = len(possible_splits)
    if (nps > msts):
        if replace:
            index_range = np.random.randint(0, nps, msts)
        else:
            index_range = np.random.choice(nps, msts, replace=False)
        return([possible_splits[i] for i in index_range])
    else:
        return(possible_splits)


def _get_possible_splits(feature_graph, feature_type, msac):
    # Query the graph structure to get the possible splits
    if (len(feature_graph.vertices) < msac):
        possible_splits = feature_graph.return_mc_partitions()
    else:
        if (feature_type == 'categorical_int') or (
                                feature_type == 'graphical_voronoi'):
            possible_splits = feature_graph.return_contracted_parts_intset(
                                  max_size_after_contraction=msac)
        else:
            possible_splits = feature_graph.return_contracted_partitions(
                                  max_size_after_contraction=msac)
    return possible_splits


cdef _score_split(g_masked, h_masked, g_sum, h_sum, gamma, reg_lambda):

    cdef double loss_score, g_left, g_right, h_left, h_right
    cdef long vec_len = len(g_masked)

    g_left = np.sum(g_masked)
    g_right = g_sum - g_left
    h_left = np.sum(h_masked)
    h_right = h_sum - h_left
    loss_score = _get_gh_score_num(g_left, g_right, h_left,
                                   h_right, gamma, reg_lambda)
    # if loss_score >= 0:
    #     loss_score = np.inf
    return loss_score


def get_mask(feature_vec_node, left_split):
    return np.array([x in left_split for x in feature_vec_node])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_mask_int_c_alt(long[::1] feature_vec_node, long[::1] left_split,
                       long vec_len, long lsplit_len,
                       cnp.ndarray[long] mask_vec):
    cdef int i, j
    for i in range(vec_len):
        for j in range(lsplit_len):
            if feature_vec_node[i] == left_split[j]:
                mask_vec[i] = 1
                break
    return mask_vec.astype(bool)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_mask_int_c(cnp.ndarray[long] feature_vec_node,
                   cnp.ndarray[long] left_split,
                   long vec_len, long lsplit_len,
                   cnp.ndarray[long] mask_vec):
    cdef int i, j
    for i in range(vec_len):
        for j in range(lsplit_len):
            if feature_vec_node[i] == left_split[j]:
                mask_vec[i] = 1
                break
    return mask_vec.astype(bool)


@cython.boundscheck(False)
@cython.wraparound(False)
def separate_indices(cnp.ndarray[long] a, cnp.ndarray[long] b,
                     cnp.ndarray[long] c, long vec_len):
    cdef long ind_a = 0, ind_b = 0, i
    for i in range(vec_len):
        if c[i] == 0:
            b[ind_b] = i
            ind_b = ind_b+1
        else:
            a[ind_a] = i
            ind_a = ind_a+1
    return a, b, ind_a


def _evaluate_feature_onehot(feature_config, feature_graph,
                             feature_vec_node, g_train_node,
                             h_train_node, gamma, reg_lambda):

    feature_type = feature_config['feature_type']
    best_split_of_feat = {}
    best_split_of_feat['loss_score'] = np.Inf
    if len(pd.unique(feature_vec_node)) <= 1:
        return(best_split_of_feat)
    g_sum = np.sum(g_train_node)
    h_sum = np.sum(h_train_node)
    feature_vals = list(feature_graph.vertices)

    # Loop within values of each feature
    for feat_val in feature_vals:
        left_split = set([feat_val])
        right_split = set(feature_vals) - left_split
        mask_left = get_mask(feature_vec_node, left_split)
        g_masked = g_train_node[mask_left]
        h_masked = h_train_node[mask_left]
        curr_loss = _score_split(g_masked, h_masked, g_sum, h_sum,
                                 gamma, reg_lambda)

        if curr_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = curr_loss
            best_split_of_feat['left_split'] = left_split
            best_split_of_feat['right_split'] = right_split
            best_split_of_feat['feature_type'] = feature_type
    return(best_split_of_feat)


def get_prediction(tree_node, X_te, dict col_to_int_dict):
    cdef cnp.ndarray[long] ind_subset_left, ind_subset_right
    cdef long vec_len, lsize
    cdef cnp.ndarray[double] next_vec

    if tree_node['node_type'] == 'leaf':
        return get_node_response_leaf(X_te.shape[0], tree_node)
    else:
        split_bool = get_node_response_df_val(X_te, tree_node, col_to_int_dict)
        vec_len = len(split_bool)
        next_vec = np.zeros(vec_len)
        ind_subset_left = np.zeros(vec_len, dtype=np.int64)
        ind_subset_right = np.zeros(vec_len, dtype=np.int64)
        ind_subset_left, ind_subset_right, lsize = separate_indices(
                                                ind_subset_left,
                                                ind_subset_right,
                                                split_bool.astype(np.int64),
                                                vec_len)
        ind_subset_left = ind_subset_left[:lsize]
        ind_subset_right = ind_subset_right[:(vec_len-lsize)]

        if lsize > 0:
            next_vec[ind_subset_left] = get_prediction(tree_node['left_child'],
                                                       X_te[
                                                       ind_subset_left, :],
                                                       col_to_int_dict)

        if lsize < vec_len:
            next_vec[ind_subset_right] = get_prediction(
                                            tree_node['right_child'],
                                            X_te[ind_subset_right, :],
                                            col_to_int_dict)
        return next_vec


def get_node_response_graphical_int(feature_vec, node):
    # fs_array = np.array(list(node['left_split'])).astype(np.int64)
    cdef int vec_len, lsplit_len

    fs_array = np.fromiter(node['left_split'], int, len(node['left_split']))
    vec_len = len(feature_vec)
    lsplit_len = len(fs_array)
    mask_vec = np.zeros(vec_len, dtype=np.int64)
    mask_vec = get_mask_int_c(feature_vec.astype(np.int64),
                              fs_array, vec_len,
                              lsplit_len, mask_vec)
    return mask_vec


def get_node_response_graphical_vor(feature_mat, node):
    tmp_tr = node['voronoi_kdtree'].query(feature_mat)
    feature_vec = tmp_tr[1]
    return get_node_response_graphical_int(feature_vec, node)


def get_node_response_graphical(feature_vec, node):
    if bool(node['na_left']):
        na_left_override = (pd.isnull(feature_vec)) & bool(node['na_left'])
        return get_mask(feature_vec, node['left_split']) | na_left_override
    else:
        return get_mask(feature_vec, node['left_split'])


def get_node_response_numerical(feature_vec, node):
    if bool(node['na_left']):
        na_left_override = (pd.isnull(feature_vec)) & bool(node['na_left'])
        return (feature_vec < node['split_val']) | na_left_override
    else:
        return (feature_vec < node['split_val'])


def get_node_response_vec(feature_vec, node):
    if node['feature_type'] == 'numerical':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return get_node_response_numerical(feature_vec, node)
    elif node['feature_type'] == 'categorical_str':
        return get_node_response_graphical(feature_vec, node)
    elif node['feature_type'] == 'categorical_int':
        return get_node_response_graphical_int(feature_vec, node)


def get_node_response_leaf(vec_len, node):
    return np.ones(vec_len)*node['node_summary_val']


def get_node_response_df_val(X_te, node, col_to_int_dict):
    if node['feature_type'] != 'graphical_voronoi':
        return get_node_response_vec(X_te[:, col_to_int_dict[
                                     node['split_feature']]], node)
    else:
        subfeature_indices = [col_to_int_dict[col_name]
                              for col_name in node['subfeature_list']]
        return get_node_response_graphical_vor(X_te[:, subfeature_indices],
                                               node)


def get_corner_mat(data_array):
    ndim = data_array.shape[1]
    max_vec = np.zeros(ndim)
    min_vec = np.zeros(ndim)
    for i in range(ndim):
        max_vec[i] = np.max(data_array[:, i])
        min_vec[i] = np.min(data_array[:, i])
    corner_mat = np.zeros((2**ndim, ndim))
    for i in range(2**ndim):
        base_num = i
        for j in range(ndim):
            ind_int = base_num//(2**(ndim-j-1))
            corner_mat[i, j] = max_vec[j] if ind_int else min_vec[j]
            base_num = base_num - ind_int*(2**(ndim-j-1))
    return corner_mat


def ridge_points_to_edge_set(rpl):
    return {frozenset([sublist[i], sublist[i+1]]) for sublist in rpl
            for i in range(-1, len(sublist) - 1)
            if (sublist[i] != -1 and sublist[i+1] != -1)}


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def max_min_hasnan(cnp.ndarray[double] my_array, long vec_len):
    cdef long i
    cdef double curr_max, curr_min, curr_val
    cdef bint has_nan = False
    curr_max = my_array[0]
    curr_min = my_array[0]
    for i in range(1, vec_len):
        curr_val = my_array[i]
        if curr_val > curr_max:
            curr_max = curr_val
        if curr_val < curr_min:
            curr_min = curr_val
        if isnan(curr_val):
            has_nan = True
    return curr_min, curr_max, has_nan
