# cython: profile=True
# cython: language_level=3

"""Decision Tree based on Discrete Graph structure"""
import graphs
import copy
import random
import warnings
import numpy as np
from scipy import spatial
import pandas as pd
from libc.math cimport log as clog
from libc.math cimport isnan
cimport numpy as np
cimport cython

ctypedef np.int64_t dtype_int64_t

class StructureDecisionTree(object):
    """Decision Tree using graphical structure.

    Uses Newton steps based on first and second derivatives of loss fn.
    """

    def __init__(self, feature_configs, min_size_split=2,
                 max_depth=3, gamma=0, feat_sample_by_node=1,
                 reg_lambda=1, feature_graphs=None):
        self.dec_tree = {}
        self.feature_configs = feature_configs
        if feature_graphs is not None:
            self.dec_tree['feature_graphs'] = feature_graphs
        else:
            self._process_feature_configs()
        self.min_size_split = min_size_split
        self.max_depth = max_depth
        self.gamma = gamma
        self.feat_sample_by_node = feat_sample_by_node
        self.reg_lambda = reg_lambda
        self.feat_list_full = list(self.feature_configs.keys())
        self.num_classes = 2
        self.optimizable = True

    def _augment_to_y_c_for_rf(self, g_h_train):
        return g_h_train

    def fit(self, X_train, g_h_train, feature_sublist=None, uv_dict=None):
        # Tree fitting works through a queue of nodes to process
        # called (node_to_proc_list)
        # The initial node is just the root of the tree
        self.num_nodes = 1
        self.node_counter = 0
        col_list = list(X_train.columns)
        self.train_column_to_int_dict = {col_list[i]: i
                                         for i in range(len(col_list))}
        self.node_to_proc_list = [self.dec_tree]

        # This next line adds a column of ones for the case where
        # we subclass to a RF Decision Tree
        g_h_train=self._augment_to_y_c_for_rf(g_h_train)

        # Initialize values to what they are at the root of the tree
        self.dec_tree['depth'] = 0
        self.dec_tree['mask'] = np.ones(g_h_train.shape[0]).astype(bool)
        if feature_sublist is not None:
            self.feature_sublist = feature_sublist
        else:
            self.feature_sublist = list(self.feature_configs.keys())
        self.X_train = X_train
        self.g_h_train = g_h_train
        if uv_dict is not None:
            self.uv_dict = uv_dict
        else:
            self.uv_dict = {}
            for feature in self.feature_configs.keys():
                if self.feature_configs[feature]['feature_type'] == 'numerical':
                    self.uv_dict[feature] = np.sort(
                                    pd.unique(X_train[feature].dropna()))


        # Process nodes until none are left to process
        while self.node_to_proc_list:
            node_to_process = self.node_to_proc_list.pop()
            self._process_tree_node(node_to_process)

        self.g_h_train = None
        self.X_train = None

    def predict(self, X_test):
        col_list = list(X_test.columns)
        column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        return self.get_prediction(self.dec_tree, X_test.to_numpy(),
                              column_to_int_dict)

    def _process_tree_node(self, curr_node):
        # Restrict to relevant data for the node in question
        X_train_node = self.X_train[curr_node['mask']]
        g_h_train_node = self.g_h_train[curr_node['mask']]

        # Save information about the current node
        num_dp = g_h_train_node.shape[0]
        curr_node['num_data_points'] = num_dp
        curr_node['node_summary_val'] = self._node_summary_gh(g_h_train_node)
        curr_node['node_index'] = self.node_counter
        self.node_counter+=1

        # Check if node is eligible to be split further
        wrap_up_now = self._check_stopping_condition(
                            num_dp, curr_node['depth'], g_h_train_node)
        if wrap_up_now:
            self._wrap_up_node(curr_node, g_h_train_node)
            return None

        features_to_search = self._get_features_to_search()
        best_split_dict = _initialize_best_split_dict()

        # Main loop over features to find best split
        for feature in features_to_search:
            best_split_for_feature = self.evaluate_feature(
                                        self.feature_configs[feature],
                                        curr_node['feature_graphs'],
                                        feature, X_train_node,
                                        g_h_train_node, self.uv_dict)
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
            self._wrap_up_node(curr_node, g_h_train_node)

    def _check_stopping_condition(self, num_dp, depth, g_h_train_node):
        cond = ((num_dp < self.min_size_split) or
                       (depth >= self.max_depth))
                       #  or
                       # (num_unique_vals<=1))
        return cond


    def _node_summary_gh(self, y_g_h_mat):
        if (y_g_h_mat.shape[0] == 0):
            return 0
        else:
            out_val = -((np.sum(y_g_h_mat[:,0]))/(np.sum(y_g_h_mat[:,1])+self.reg_lambda))
            return(out_val)

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

    def _wrap_up_node(self, curr_node, g_h_train_node):
        # Compute summary stats of node and mark it as a leaf
        # curr_node['node_summary_val'] = self._node_summary_gh(g_h_train_node)
        curr_node['num_data_points'] = g_h_train_node.shape[0]
        curr_node['node_type'] = 'leaf'
        curr_node.pop('mask')

    def _execute_split(self, curr_node, best_split_dict, feature_graphs_node):
        self.num_nodes+=2
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
            warnings.warn("Unknown feature type")

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
            left_mask = (self.X_train[curr_feat] <= split_val).to_numpy()

        curr_node['split_val'] = split_val
        curr_node['loss_score'] = best_split_dict['loss_score']
        split_feat = best_split_dict['split_feature']
        curr_node['split_feature'] = split_feat
        curr_node['train_feat_col'] = self.train_column_to_int_dict[split_feat]
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

        if best_split_dict['feature_type']=='categorical_str':
            self.optimizable = False
        feat_vec_train = self.X_train[best_split_dict['split_feature']]
        left_mask = feat_vec_train.isin(
                        best_split_dict['left_split']).to_numpy()
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
        split_feat = best_split_dict['split_feature']
        curr_node['split_feature'] = split_feat
        curr_node['train_feat_col'] = self.train_column_to_int_dict[split_feat]
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

        self.optimizable = False
        subfeature_indices = [self.train_column_to_int_dict[colname]
                              for colname in best_split_dict[
                                                'subfeature_list']]
        data_array = self.X_train.to_numpy()[:, subfeature_indices]
        num_query_pts = data_array.shape[0]
        num_vor_dims = data_array.shape[1]
        feat_vec = np.zeros(num_query_pts,dtype=np.int64)
        feat_vec = map_to_nn_point_index(best_split_dict['vor_pts'],
                                         data_array,
                                         feat_vec,
                                         best_split_dict['vor_pts'].shape[0],
                                         num_query_pts, num_vor_dims)
        fs_array = np.fromiter(best_split_dict['left_split'], int,
                               len(best_split_dict['left_split'])).astype(np.int64)
        vec_len = len(feat_vec)
        lsplit_len = len(fs_array)
        left_mask = np.zeros(vec_len, dtype=np.int64)
        left_mask = get_mask_int_c(feat_vec, fs_array, vec_len,
                                   lsplit_len, left_mask)

        # record info about current node
        curr_node['left_split'] = best_split_dict['left_split']
        curr_node['right_split'] = (best_split_dict['voronoi_graph'].vertices -
                                    best_split_dict['left_split'])
        curr_node['num_voronoi_edges'] = len(best_split_dict['voronoi_graph'].edges)
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']
        sub_feat_list = best_split_dict['subfeature_list']
        curr_node['subfeature_list'] = sub_feat_list
        sub_feat_cols = np.array([self.train_column_to_int_dict[col] for col in sub_feat_list])
        curr_node['subfeature_cols'] = sub_feat_cols
        curr_node['vor_pts'] = best_split_dict['vor_pts']
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        curr_mask = curr_node.pop('mask')

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_right = feature_graphs_node.copy()

        self._create_children_nodes(curr_node, feature_graphs_left,
                                    feature_graphs_right, curr_mask, left_mask)

    def evaluate_feature(self, feature_config, feature_graphs, feature_name,
                         X_train_node, g_h_train_node, uv_dict):
        ft = feature_config['feature_type']

        if ft == 'numerical':
            return self._evaluate_feature_numerical(
                            feature_config, X_train_node[feature_name].to_numpy(),
                            g_h_train_node, uv_dict[feature_name])
        elif ((ft == 'categorical_int') or (ft == 'categorical_str')):
            split_method = feature_config['split_method']
            if split_method == 'contraction':
                return self._evaluate_feature_enum_contr(
                            feature_config, feature_graphs[feature_name],
                            X_train_node[feature_name].to_numpy(),
                            g_h_train_node)
            elif split_method == 'span_tree':
                return self._evaluate_feature_multitree(
                            feature_config, feature_graphs[feature_name],
                            X_train_node[feature_name].to_numpy(),
                            g_h_train_node)
            elif split_method == 'onehot':
                return self._evaluate_feature_onehot(
                                feature_config, feature_graphs[feature_name],
                                X_train_node[feature_name].to_numpy(),
                                g_h_train_node)
            else:
                w_str = 'Unknown split method "{}" - ignoring feature'.format(
                                                                    split_method)
                warnings.warn(w_str)
        elif ft == 'graphical_voronoi':
            return self._evaluate_feature_voronoi(
                            feature_config, X_train_node,
                            g_h_train_node)
        else:
            warnings.warn('Unknown feature type "{}": ignoring'.format(ft))


    def _evaluate_feature_numerical(self, feature_config, feature_vec,
                                    g_h_mat, uv_array):
        interp_mode = ('random' if 'interp_mode' not in feature_config.keys()
                       else feature_config['interp_mode'])
        # May be slightly more efficient to reduce to number of splits needed
        # and then calculate them. (Don't have to interpolate as much)
        splits_to_eval = _get_numerical_splits(feature_vec, uv_array,
                                               interp_mode=interp_mode)
        if len(splits_to_eval) == 0:
            return(_initialize_best_split_dict())
        else:
            splits_to_eval = _subset_splits(splits_to_eval, feature_config)

            best_loss, best_spl_val, na_left, na_rnd = self._evaluate_numerical_splits(
                                            feature_vec, g_h_mat,
                                            splits_to_eval)
            best_split_of_feat = {}
            best_split_of_feat['loss_score'] = best_loss
            best_split_of_feat['left_split'] = best_spl_val
            best_split_of_feat['feature_type'] = 'numerical'
            best_split_of_feat['na_left'] = na_left
            best_split_of_feat['na_dir_random'] = na_rnd
            return(best_split_of_feat)


    def _evaluate_numerical_splits(self, feature_vec, g_h_mat,
                                   split_vec):

        has_na_vals = np.isnan(split_vec[-1])
        bin_result_vec = np.searchsorted(split_vec,
                                         feature_vec,
                                         side='right').astype(np.int64)
        g_sum_bins, h_sum_bins = get_bin_sums_c(g_h_mat,
                                                bin_result_vec,
                                                len(split_vec)+1)
        g_sum_total, g_sum_left, g_sum_right = get_left_right_sums(g_sum_bins)
        h_sum_total, h_sum_left, h_sum_right = get_left_right_sums(h_sum_bins)
        score_vec = (-1)*_get_gh_score_array(g_sum_left, g_sum_right,
                                             h_sum_left, h_sum_right,
                                             self.gamma, self.reg_lambda)

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
                                                     self.gamma, self.reg_lambda)
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


    def _evaluate_feature_enum_contr(self, feature_config, feature_graph,
                                     feature_vec_node, g_h_train_node):

        possible_splits = _get_possible_splits(feature_graph,
                                               feature_config)

        splits_to_eval = _choose_split_subset(possible_splits,
                                              feature_config[
                                                     'max_splits_to_search'])

        best_split_of_feat = self._eval_split_set(feature_vec_node, g_h_train_node,
                                             splits_to_eval,
                                             feature_config['feature_type'])
        return(best_split_of_feat)


    def _eval_split_set(self, feature_vec_node, g_h_train_node,
                        splits_to_eval, feature_type):

        best_split_of_feat = _initialize_best_split_dict()
        g_h_sum = np.sum(g_h_train_node, axis=0)
        is_integer_valued = (feature_type == 'categorical_int') or (
                             feature_type == 'graphical_voronoi')

        # Loop within values of each feature
        for partition in splits_to_eval:
            curr_partition = list(partition)
            left_split = curr_partition[0]
            right_split = curr_partition[1]
            if is_integer_valued:
                fs_array = np.fromiter(left_split, int,
                                       len(left_split)).astype(np.int64)
                vec_len = len(feature_vec_node)
                lsplit_len = len(fs_array)
                mask_left = np.zeros(vec_len, dtype=np.int64)
                mask_left = get_mask_int_c(feature_vec_node.astype(np.int64),
                                               fs_array, vec_len, lsplit_len,
                                               mask_left)
            else:
                mask_left = get_mask(feature_vec_node, left_split)

            curr_loss = self.get_score_of_split(g_h_train_node, mask_left, g_h_sum)

            if curr_loss < best_split_of_feat['loss_score']:
                best_split_of_feat['loss_score'] = curr_loss
                best_split_of_feat['left_split'] = left_split
                best_split_of_feat['feature_type'] = feature_type
                # Next two lines may be unnecessary...
                best_split_of_feat['na_left'] = int(random.random() < .5)
                best_split_of_feat['na_dir_random'] = 1


        return(best_split_of_feat)

    def get_score_of_split(self, g_h_train_node, mask_left, g_h_sum):
        g_h_masked = g_h_train_node[mask_left,:]
        g_masked = g_h_masked[:,0]
        h_masked = g_h_masked[:,1]
        curr_loss = _score_split(g_masked, h_masked, g_h_sum[0], g_h_sum[1],
                                 self.gamma, self.reg_lambda)
        return(curr_loss)


    def _evaluate_feature_voronoi(self, feature_config, X_train_node, g_h_train_node):

        # pull and process config items
        feature_type = feature_config['feature_type']
        sub_features = feature_config['subfeature_list']
        vor_sample_size = feature_config['voronoi_sample_size']
        vor_min_sample_size = feature_config['voronoi_min_pts']
        force_complete = (('force_complete' in feature_config.keys())
                         and feature_config['force_complete'])
        if (X_train_node.shape[0] < vor_min_sample_size):
            return(_initialize_best_split_dict())
        vor_sample_size = np.minimum(vor_sample_size, X_train_node.shape[0])

        feature_graph, vor_pts, data_array = _get_graph_vor_pts(X_train_node,
                                                                   sub_features,
                                                                   vor_sample_size,
                                                                   force_complete)
        if feature_graph is None:
            return({})
        # Map data points to regions

        num_query_pts = data_array.shape[0]
        num_vor_dims = data_array.shape[1]
        feature_vec_node = np.zeros(num_query_pts,dtype=np.int64)
        feature_vec_node = map_to_nn_point_index(vor_pts,
                                         data_array,
                                         feature_vec_node,
                                         vor_pts.shape[0],
                                         num_query_pts, num_vor_dims)
     
        if feature_config['split_method'] == 'span_tree':
            best_split_of_feat = self._evaluate_feature_multitree(feature_config,
                                                             feature_graph,
                                                             feature_vec_node,
                                                             g_h_train_node)
        elif feature_config['split_method'] == 'contraction':
            best_split_of_feat = self._evaluate_feature_enum_contr(feature_config,
                                                              feature_graph,
                                                              feature_vec_node,
                                                              g_h_train_node)
        else:
            warnings.warn('Unknown method for splitting feature')

        best_split_of_feat['vor_pts'] = vor_pts
        best_split_of_feat['voronoi_graph'] = feature_graph
        best_split_of_feat['subfeature_list'] = sub_features
        return(best_split_of_feat)


    def _evaluate_feature_multitree(self, feature_config, feature_graph,
                                    feature_vec_node,
                                    g_h_train_node):

        if not feature_graph.vertices:
            return(_initialize_best_split_dict())

        g_h_sum, g_h_val_arr, map_dict = self._init_for_span_trees(
                                                    feature_vec_node,
                                                    g_h_train_node,
                                                    feature_graph,
                                                    feature_config['feature_type'])

        best_split_of_feat = self._eval_multiple_span_trees(feature_vec_node,
                                                       g_h_train_node,
                                                       g_h_sum,
                                                       g_h_val_arr,
                                                       map_dict,
                                                       feature_config[
                                                        'num_span_trees'],
                                                       feature_graph,
                                                       feature_config[
                                                        'feature_type'])
        return(best_split_of_feat)


    def _init_for_span_trees(self, feature_vec_node, g_h_train_node,
                             feature_graph, feature_type):
        g_h_sum = np.sum(g_h_train_node, axis=0)
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

        if not is_integer_valued:
            fvn_int = np.array([map_dict[i] for i in feature_vec_node])
            g_h_val_arr = self.get_gh_val_mat(fvn_int,g_h_train_node,
                                  max_num_vertices)
        else:
            g_h_val_arr = self.get_gh_val_mat(feature_vec_node,g_h_train_node,
                                          max_num_vertices)

        return g_h_sum, g_h_val_arr, map_dict

    def get_gh_val_mat(self, feature_vec_node,g_h_train_node,
                       max_num_vertices):
        g_h_val_arr = np.zeros((max_num_vertices,2))
        g_h_val_arr = get_g_h_feature_sum_arrays(
                                            feature_vec_node.astype(np.int64),
                                            g_h_train_node,
                                            g_h_val_arr)
        return g_h_val_arr


    def _eval_multiple_span_trees(self, feature_vec_node, g_h_train_node, g_h_sum,
                                  g_h_val_arr, map_dict,
                                  num_spanning_trees, feature_graph, feature_type):
        best_split_of_feat = _initialize_best_split_dict()
        for i in range(num_spanning_trees):
            g_h_accum_array = g_h_val_arr.copy()
            curr_span_tree, drd = feature_graph.get_uniform_random_spanning_tree()
            root_dist_list = [(a, b) for a, b in drd.items()]
            vertex_arr = np.array([x[0] for x in root_dist_list])
            dist_arr = np.array([x[1] for x in root_dist_list])
            vertex_order = vertex_arr[np.argsort(-dist_arr)]
            vertex_to_split_dict = {vertex: set([vertex])
                                    for vertex in curr_span_tree.vertices}
            best_split_of_feat = self._eval_span_tree(feature_vec_node,g_h_train_node,
                                                 curr_span_tree, g_h_accum_array,
                                                 g_h_sum,
                                                 best_split_of_feat,
                                                 vertex_to_split_dict,
                                                 feature_type,
                                                 vertex_order, map_dict)
        return(best_split_of_feat)


    def _eval_span_tree(self, feature_vec_node, g_h_train_node, span_tree,
                        g_h_accum_array, g_h_sum,
                        best_split_of_feat, vertex_to_split_dict,
                        feature_type,
                        vertex_order, map_dict=None):

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

            left_feat_values = vertex_to_split_dict[leaf_vertex_raw]
            curr_loss = self.get_loss_in_span_tree(feature_vec_node, g_h_train_node,
                                                   g_h_accum_array,
                                                   g_h_sum,
                                                   leaf_vertex_ind, left_feat_values,
                                                   feature_type)
            left_prob = .5
            na_left, na_rnd = int(random.random() < left_prob), 1
            curr_loss_nal = np.Inf

            if curr_loss < best_split_of_feat['loss_score']:
                best_split_of_feat['loss_score'] = curr_loss
                best_split_of_feat['left_split'] = vertex_to_split_dict[
                                                        leaf_vertex_raw]
                best_split_of_feat['feature_type'] = feature_type
                best_split_of_feat['na_left'] = na_left
                best_split_of_feat['na_dir_random'] = na_rnd

            g_h_accum_array = self.update_g_h_accum(g_h_accum_array,
                                            leaf_neighbor_ind,
                                            leaf_vertex_ind)
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


    def get_loss_in_span_tree(self, feature_vec_node, g_h_train_node,
                              g_h_accum_array, g_h_sum,
                              leaf_vertex_ind, left_feat_values,
                                                   feature_type):
        g_left = g_h_accum_array[leaf_vertex_ind,0]
        h_left = g_h_accum_array[leaf_vertex_ind,1]
        curr_loss = _get_gh_score_num(g_left, g_h_sum[0]-g_left,
                                      h_left, g_h_sum[1]-h_left,
                                      self.gamma, self.reg_lambda)
        return(curr_loss)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def update_g_h_accum(self, np.ndarray[double, ndim=2] g_h_accum_array,
        long lni, long lvi):
        g_h_accum_array[lni,0] +=  g_h_accum_array[lvi,0]
        g_h_accum_array[lni,1] += g_h_accum_array[lvi,1]
        return g_h_accum_array

    def _evaluate_feature_onehot(self, feature_config, feature_graph,
                                 feature_vec_node, g_h_train_node):

        feature_type = feature_config['feature_type']
        best_split_of_feat = {}
        best_split_of_feat['loss_score'] = np.Inf
        if len(pd.unique(feature_vec_node)) <= 1:
            return(best_split_of_feat)
        g_h_sum = np.sum(g_h_train_node, axis=0)
        feature_vals = list(feature_graph.vertices)

        # Loop within values of each feature
        for feat_val in feature_vals:
            left_split = set([feat_val])
            right_split = set(feature_vals) - left_split
            mask_left = get_mask(feature_vec_node, left_split)
            gh_masked = g_h_train_node[mask_left,:]
            curr_loss = self.get_score_of_split(g_h_train_node, mask_left, g_h_sum)

            if curr_loss < best_split_of_feat['loss_score']:
                best_split_of_feat['loss_score'] = curr_loss
                best_split_of_feat['left_split'] = left_split
                best_split_of_feat['right_split'] = right_split
                best_split_of_feat['feature_type'] = feature_type
        return(best_split_of_feat)


    def get_prediction(self, tree_node, X_te, dict col_to_int_dict):
        cdef np.ndarray[dtype_int64_t] ind_subset_left, ind_subset_right
        cdef long vec_len, lsize
        cdef np.ndarray[double] next_vec

        if tree_node['node_type'] == 'leaf':
            return get_node_response_leaf(X_te.shape[0], tree_node)
        else:
            split_bool = get_node_response_df_val(X_te, tree_node, col_to_int_dict)
            vec_len = len(split_bool)
            next_vec = np.zeros(vec_len)
            ind_subset_left = np.empty(vec_len, dtype=np.int64)
            ind_subset_right = np.empty(vec_len, dtype=np.int64)
            ind_subset_left, ind_subset_right, lsize = separate_indices(
                                                    ind_subset_left,
                                                    ind_subset_right,
                                                    split_bool.astype(np.int64),
                                                    vec_len)
            ind_subset_left = ind_subset_left[:lsize]
            ind_subset_right = ind_subset_right[:(vec_len-lsize)]

            if lsize > 0:
                next_vec[ind_subset_left] = self.get_prediction(tree_node['left_child'],
                                                           X_te[
                                                           ind_subset_left, :],
                                                           col_to_int_dict)

            if lsize < vec_len:
                next_vec[ind_subset_right] = self.get_prediction(
                                                tree_node['right_child'],
                                                X_te[ind_subset_right, :],
                                                col_to_int_dict)
            return next_vec


    def get_max_split_size(self):
        return(get_max_cat_left_split(self.dec_tree))


    def _process_feature_configs(self):
        fg = {}
        for feat_name in self.feature_configs.keys():
            if 'graph' in self.feature_configs[feat_name].keys():
                fg[feat_name] = self.feature_configs[feat_name]['graph']
            if 'split_method' in self.feature_configs[feat_name].keys():
                if self.feature_configs[feat_name]['split_method'] == 'onehot':
                    fg[feat_name] = graphs.complete_graph(
                                    self.feature_configs[
                                                 feat_name]['feature_vals'])
        self.feature_graphs = fg


def get_max_cat_left_split(node):
    if node['node_type']=='leaf':
        return(0)
    else:
        if node['feature_type']=='categorical_int':
            return(np.max([len(node['left_split']), 
                           get_max_cat_left_split(node['left_child']),
                           get_max_cat_left_split(node['right_child'])]))
        else:
            return(np.max([get_max_cat_left_split(node['left_child']),
                           get_max_cat_left_split(node['right_child'])]))

def _initialize_best_split_dict():
    out_dict = {}
    out_dict['loss_score'] = np.inf
    out_dict['left_split'] = None
    out_dict['split_feature'] = None
    return(out_dict)


def _get_gh_score_num(double g_left,  double g_right,
                      double h_left, double h_right,
                      double gamma, double reg_lambda, double tol=1e-12):
    loss_val = -1.0 * (.5*(((g_left*g_left)/(h_left+reg_lambda)) +
                       ((g_right*g_right)/(h_right+reg_lambda)) -
                   (((g_left + g_right)*(g_left + g_right)) /
                    (h_left + h_right+reg_lambda)))-gamma)
    if loss_val >= -tol:
        loss_val = np.inf
    return(loss_val)


def _get_gh_score_array(np.ndarray[double] g_left,
                        np.ndarray[double] g_right,
                        np.ndarray[double] h_left,
                        np.ndarray[double] h_right,
                        double gamma, double reg_lambda):
    return(.5*(((g_left*g_left)/(h_left+reg_lambda)) +
               ((g_right*g_right)/(h_right+reg_lambda)) -
               (((g_left+g_right) * (g_left+g_right)) /
                (h_left+h_right+reg_lambda)))-gamma)



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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_val, max_val, has_na = max_min_hasnan(feature_vec.astype(float),
                                                  len(feature_vec))
        unique_vals = _get_unique_vals(uv_array, min_val, max_val)
        if has_na and not np.isnan(max_val):
            unique_vals = np.concatenate((unique_vals, [np.nan]))
    if len(unique_vals) <= 1:
        return []
    else:
        weight = .5 if interp_mode == 'mid' else random.random()
        splits = (weight * unique_vals[1:]) + ((1-weight) * unique_vals[:-1])
        return splits

def _get_unique_vals(uv_array, min_val, max_val):
    return(uv_array[(uv_array >= min_val) & (
                                uv_array <= max_val)])


def _get_best_vals(score_vec, split_vec):
    best_split_index = np.argmin(score_vec)
    # best_loss = np.min(score_vec)
    best_loss = score_vec[best_split_index]
    best_split_val = split_vec[best_split_index]
    return best_loss, best_split_val


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_bin_sums_c(np.ndarray[double, ndim=2] g_h_mat,
                   np.ndarray[dtype_int64_t] bin_result_vec, long out_vec_size):
    cdef int i
    cdef int m = bin_result_vec.shape[0]

    cdef np.ndarray[double] g_sum_bins = np.zeros(out_vec_size)
    cdef np.ndarray[double] h_sum_bins = np.zeros(out_vec_size)

    for i in range(m):
        g_sum_bins[bin_result_vec[i]] += g_h_mat[i,0]
        h_sum_bins[bin_result_vec[i]] += g_h_mat[i,1]
    return g_sum_bins, h_sum_bins



def get_left_right_sums_vec(bin_sums):
    sum_left = np.cumsum(bin_sums, axis=0)
    sum_total = sum_left[-1,:]
    sum_left = sum_left[:-1,:]
    sum_right = sum_total - sum_left
    return sum_total, sum_left, sum_right


def get_left_right_sums(bin_sums):
    sum_left = np.cumsum(bin_sums)
    sum_total = sum_left[-1]
    sum_left = sum_left[:-1]
    sum_right = sum_total - sum_left
    return sum_total, sum_left, sum_right




def _get_graph_vor_pts(X_train_node, sub_features, vor_sample_size,
                        force_complete):
    data_array = get_data_array(X_train_node, sub_features)
    cm = get_corner_mat(data_array)
    npts, ndim = data_array.shape
    voronoi_sample_mat = sample_voronoi_pts(data_array, npts, vor_sample_size)
    if ndim <= 4:
        voronoi_sample_mat = np.concatenate((voronoi_sample_mat, cm), axis=0)
    else:
        num_add_corners = np.maximum(ndim, 10)
        inds = np.random.choice(cm.shape[0], num_add_corners, replace=False)
        cm = cm[inds, :]
        voronoi_sample_mat = np.concatenate((voronoi_sample_mat, cm), axis=0)
    voronoi_sample_mat = np.unique(voronoi_sample_mat, axis=0)
    if voronoi_sample_mat.shape[0]<4:
        return(None,None,None)
    if not force_complete:
        vor_obj = get_voronoi_obj(voronoi_sample_mat)
        feature_graph = graphs.graph_undirected(
                            ridge_points_to_edge_set(vor_obj.ridge_points))
    else:
        feature_graph=graphs.complete_int_graph(0,voronoi_sample_mat.shape[0])
    return feature_graph, voronoi_sample_mat, data_array


def sample_voronoi_pts(data_array, npts, vor_sample_size):
    return(data_array[np.random.randint(0, npts, vor_sample_size),:])

def get_data_array(X_train_node, sub_features):
    return(X_train_node.loc[:, sub_features].to_numpy())

def get_ckd_tree(voronoi_sample_mat):
    return(spatial.cKDTree(voronoi_sample_mat))

def get_voronoi_obj(voronoi_sample_mat):
    return(spatial.Voronoi(voronoi_sample_mat, qhull_options="Qbb Qc Qz QJ"))






@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_g_h_feature_sum_arrays(np.ndarray[dtype_int64_t] feature_vec_node,
                               np.ndarray[double, ndim=2] g_h_train_node,
                               np.ndarray[double, ndim=2] g_h_val_arr):
    cdef long i, ind
    cdef long array_size = len(feature_vec_node)
    for i in range(array_size):
        ind = feature_vec_node[i]
        g_h_val_arr[ind,0] = g_h_val_arr[ind,0]+g_h_train_node[i,0]
        g_h_val_arr[ind,1] = g_h_val_arr[ind,1]+g_h_train_node[i,1]
    return g_h_val_arr


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# def get_g_h_feature_sum_arrays_gen(feature_vec_node, dict mapping_dict,
#                                    np.ndarray[double, ndim=2] g_h_train_node,
#                                    np.ndarray[double, ndim=2] g_h_val_arr):
#     cdef long i, ind
#     cdef long array_size = len(feature_vec_node)
#     for i in range(array_size):
#         ind = mapping_dict[feature_vec_node[i]]
#         g_h_val_arr[ind,0] = g_h_val_arr[ind,0]+g_h_train_node[i,0]
#         g_h_val_arr[ind,1] = g_h_val_arr[ind,1]+g_h_train_node[i,1]
#     return g_h_val_arr







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


def _get_possible_splits(feature_graph, feature_config):
    # Query the graph structure to get the possible splits
    msac = feature_config['contraction_size']
    feature_type = feature_config['feature_type']
    if 'edge_method' in feature_config.keys():
        esm = feature_config['edge_method']
        feature_config['edge_method_used'] = esm
    else:
        esm = 'random_edge'
    if (len(feature_graph.vertices) < msac):
        possible_splits = feature_graph.return_mc_partitions()
    else:
        if (feature_type == 'categorical_int') or (
                                feature_type == 'graphical_voronoi'):
            possible_splits = feature_graph.return_contracted_parts_intset(
                                  max_size_after_contraction=msac)
        else:
            possible_splits = feature_graph.return_contracted_partitions(
                                  max_size_after_contraction=msac,
                                  edge_selection=esm)
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

# def get_mask(feature_vec_node, left_split):
#     return np.isin(feature_vec_node,list(left_split))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_mask_int_c(np.ndarray[dtype_int64_t] feature_vec_node,
                   np.ndarray[dtype_int64_t] left_split,
                   long vec_len, long lsplit_len,
                   np.ndarray[dtype_int64_t] mask_vec):
    cdef int i, j
    for i in range(vec_len):
        for j in range(lsplit_len):
            if feature_vec_node[i] == left_split[j]:
                mask_vec[i] = 1
                break
    return mask_vec.astype(bool)


@cython.boundscheck(False)
@cython.wraparound(False)
def separate_indices(np.ndarray[dtype_int64_t] a, np.ndarray[dtype_int64_t] b,
                     np.ndarray[dtype_int64_t] c, long vec_len):
    cdef long ind_a = 0, ind_b = 0, i
    for i in range(vec_len):
        if c[i] == 0:
            b[ind_b] = i
            ind_b = ind_b+1
        else:
            a[ind_a] = i
            ind_a = ind_a+1
    return a, b, ind_a






def get_node_response_graphical_int(feature_vec, node):
    # fs_array = np.array(list(node['left_split'])).astype(np.int64)
    cdef int vec_len, lsplit_len

    fs_array = np.fromiter(node['left_split'], int,
                           len(node['left_split'])).astype(np.int64)
    vec_len = len(feature_vec)
    lsplit_len = len(fs_array)
    mask_vec = np.zeros(vec_len, dtype=np.int64)
    mask_vec = get_mask_int_c(feature_vec.astype(np.int64),
                              fs_array, vec_len,
                              lsplit_len, mask_vec)
    return mask_vec


def get_node_response_graphical_vor(feature_mat, node):
    num_query_pts = feature_mat.shape[0]
    num_vor_dims = feature_mat.shape[1]
    feature_vec = np.zeros(num_query_pts,dtype=np.int64)
    feature_vec = map_to_nn_point_index(node['vor_pts'],
                                        feature_mat,
                                        feature_vec,
                                        node['vor_pts'].shape[0],
                                        num_query_pts, num_vor_dims)
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
    if type(node['node_summary_val'])!=np.ndarray:
        return np.ones(vec_len)*node['node_summary_val']
    else:
        return(np.tile(node['node_summary_val'], (vec_len,1)))


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
    max_vec = np.max(data_array, axis=0)
    min_vec = np.min(data_array, axis=0)
    corner_mat = np.zeros((2**ndim, ndim))
    for i in range(2**ndim):
        base_num = i
        for j in range(ndim):
            ind_int = base_num//(2**(ndim-j-1))
            corner_mat[i, j] = max_vec[j] if ind_int else min_vec[j]
            base_num = base_num - ind_int*(2**(ndim-j-1))
    return corner_mat


def ridge_points_to_edge_set(rpl):
    return(rpl[np.min(rpl, axis=1)>=0])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def map_to_nn_point_index(double[:,:] core_pts, double[:,:] query_pts,
                          np.ndarray[dtype_int64_t] out_mat, long num_core_pts,
                          long num_query_pts, long ndim):
    cdef double curr_dist, tnum, best_dist
    cdef int i,j, best_j
    for i in range(num_query_pts):
        best_j=0
        best_dist=1e16
        for j in range(num_core_pts):
            curr_dist = 0           
            for k in range(ndim):
                tnum = (query_pts[i,k] - core_pts[j,k])
                curr_dist += tnum*tnum
                # if curr_dist>best_dist:
                #     break
            if curr_dist<best_dist:
                best_dist = curr_dist
                best_j = j
        out_mat[i] = best_j
    return(out_mat)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def max_min_hasnan(np.ndarray[double] my_array, long vec_len):
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


from copy import deepcopy

def ts_extend(m_list, pz, po, feature_name):
    m_len = len(m_list)
    new_m = deepcopy(m_list)
    new_m.append({})
    new_m[m_len]['feature_name'] = feature_name
    new_m[m_len]['z_info'] = pz
    new_m[m_len]['o_info'] = po
    new_m[m_len]['w_info'] = 1 if m_len==0 else 0
    for i in range(m_len-1,-1,-1):
        new_m[i+1]['w_info'] += po*new_m[i]['w_info']*(i+1)/(m_len+1)
        new_m[i]['w_info'] = pz*new_m[i]['w_info']*(m_len-i)/(m_len+1)
    return new_m


def ts_unwind(m_list, i):
    m_len = len(m_list)
    n = m_list[m_len-1]['w_info']
    new_m = deepcopy(m_list)
    for j in range(m_len-1,0,-1):
        if m_list[i-1]['o_info']!=0:
            t = new_m[j-1]['w_info']
            new_m[j-1]['w_info'] = n*m_len/(j*m_list[i-1]['o_info'])
            n = t - new_m[j-1]['w_info']*m_list[i-1]['z_info']*((m_len-j)/m_len)
        else:
            new_m[j-1]['w_info'] = (new_m[j-1]['w_info'] * m_len)/(m_list[i-1]['z_info']*(m_len - j))
    if ((i-1)<=(m_len-1)):
        for j in range(i-1, m_len-1):
            new_m[j-1]['feature_name'] = new_m[j]['feature_name']
            new_m[j-1]['z_info'] = new_m[j]['z_info']
            new_m[j-1]['o_info'] = new_m[j]['o_info']
    return new_m

def ts_unwind_2(m_list, i):
    m_len = len(m_list)
    n = m_list[m_len-1]['w_info']
    for j in range(m_len-1,0,-1):
        if m_list[i-1]['o_info']!=0:
            t = m_list[j-1]['w_info']
            m_list[j-1]['w_info'] = n*m_len/(j*m_list[i-1]['o_info'])
            n = t - m_list[j-1]['w_info']*m_list[i-1]['z_info']*((m_len-j)/m_len)
        else:
            m_list[j-1]['w_info'] = (m_list[j-1]['w_info'] * m_len)/(m_list[i-1]['z_info']*(m_len - j))
    if ((i-1)<=(m_len-1)):
        for j in range(i-1, m_len-1):
            m_list[j-1]['feature_name'] = m_list[j]['feature_name']
            m_list[j-1]['z_info'] = m_list[j]['z_info']
            m_list[j-1]['o_info'] = m_list[j]['o_info']
#     return new_m


def ts_unwind_sum(m_list, i):
    m_len = len(m_list)
    n = m_list[m_len-1]['w_info']
    out_val = 0
    for j in range(m_len-1,0,-1):
        if m_list[i-1]['o_info']!=0:
            tmp = n*m_len/(j*m_list[i-1]['o_info'])
            out_val+=tmp
            n = m_list[j-1]['w_info'] - tmp*m_list[i-1]['z_info']*((m_len-j)/m_len)
        else:
            tmp = (m_list[j-1]['w_info'] * m_len)/(m_list[i-1]['z_info']*(m_len - j))
            out_val += (m_list[j-1]['w_info'] * m_len)/(m_list[i-1]['z_info']*(m_len - j))
    return out_val




def ts_recurse(phi_dict, curr_node, data_point, m_list, pz, po, sf='intercept'):
    m_list = ts_extend(m_list, pz, po, sf)
    if curr_node['node_type']!= 'interior':
        # print(m_list)
        # print(phi_dict)
        for i in range(2, len(m_list)+1):
            temp_m = ts_unwind(m_list,i)
            w = ts_unwind_sum(m_list,i)
            # w = np.sum(np.array([temp_m[k]['w_info'] for k in range(len(temp_m)-1)]))         
            increment = w*(m_list[i-1]['o_info']-m_list[i-1]['z_info'])*curr_node['node_summary_val']
            phi_dict[m_list[i-1]['feature_name']] = phi_dict[m_list[i-1]['feature_name']] + increment
    else:
        gl = go_left(data_point, curr_node)
        if gl:
            h = curr_node['left_child']
            c = curr_node['right_child']
        else:
            c = curr_node['left_child']
            h = curr_node['right_child']
        iz, io = 1,1
        feature_path = [m_list[q]['feature_name'] for q in range(len(m_list))]
        first_occ = next((q for q in range(len(feature_path)-1) if feature_path[q]==curr_node['split_feature']), np.nan)
        if not (np.isnan(first_occ)):
            iz, io = m_list[first_occ]['z_info'], m_list[first_occ]['o_info']
            # print('unwinding repeat')
            # m_list = ts_unwind(m_list,first_occ)
            ts_unwind_2(m_list)
        rh = h['num_data_points']
        rc = c['num_data_points']
        rj = curr_node['num_data_points']
        sf = curr_node['split_feature']
        ts_recurse(phi_dict, h, data_point, m_list, iz*(rh/rj), io, sf)
        ts_recurse(phi_dict, c, data_point, m_list, iz*(rc/rj), 0, sf) # not sure if last arg should be 0 or io

def single_tree_shap(data_point, curr_tree, feature_list):
    phi_dict = {feature_list[i]:0 for i in range(len(feature_list))}
    phi_dict['intercept']=curr_tree['node_summary_val']
    ts_recurse(phi_dict, curr_tree, data_point, [], 1, 1)
    return phi_dict

def go_left(data_point, curr_node):
    if curr_node['feature_type']=='numerical':
        return(data_point[curr_node['split_feature']]<curr_node['split_val'])
    if (curr_node['feature_type'] in ['categorical_str', 'categorical_int']):
        return(data_point[curr_node['split_feature']] in curr_node['left_split'])


