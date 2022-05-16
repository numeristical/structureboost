# cython: profile=True
# cython: language_level=3

"""Structured Random Forest using graphs"""
import warnings
import numpy as np
import pandas as pd
import structure_rfdt as strfdt
import graphs
import random
from libc.math cimport log as clog
from libc.math cimport exp
cimport numpy as np
cimport cython


class StructureRF(object):
    """Random Forest model allowing categorical structure.

    Requires a feature-specific configuration -- see docs and examples:
        www.github.com/numeristical/structureboost/examples

    Categorical variables require a graph as part of their feature config.

    Parameters
    ----------

    num_trees : integer, greater than zero
        The (maximum) number of trees to build in fitting the model.  We
        highly recommend using early stopping rather than guesswork or
        grid search to set this parameter.

    feature_configs : dict
        A dictionary that contains the specifications for each feature in
        the models.  StructureBoost allows an unusual amount of flexibility
        which necessitates a more complicated configuration. See
        https://structureboost.readthedocs.io/en/latest/feat_config.html
        for details.

    mode : 'classification' or 'regression', default is 'classification'
        Whether or not to treat this as a classification or a regression
        problem.  If 'classification', log-loss (aka cross-entropy)
        will be the default loss function, for 'regression', the default
        loss function will be mean squared error.

    max_depth : int, default is 25
        The maximum depth to use when building trees. Usually for random
        forests, it is best to have a high max_depth and control tree
        size via min_size_split.

    subsample : float, default='1'
        The size of the data set used to build each tree, expressed as a
        fraction of the training set size. By default this will choose a
        sample the size of the training set.  A float between 0 and 1 will
        indicate using a smaller data sample. See also `replace`.

    replace : bool, default=True
        Whether or not to use replacement when choosing the training
        sample for each tree.  Default is True, meaning it will choose
        a "bootstrap" style sample where multiple instances of the
        same data point are potentially present.

    min_size_split : int, default is 2
        The minimum size required for a node to be split.  Any node
        smaller than this will not be considered for a split.

    feat_sample_by_tree : float, default is 1
        The percentage of features to consider for each tree.
        Works multiplicatively with feat_ample_by_node.

    feat_sample_by_node : float, default is 1
        The percentage of remaining features to consider for each node.
        This will sample from the features remaining after the
        features are sampled for each tree.

    na_unseen_action : 'random', 'weighted_random', default is 'weighted_random'
        How to determine NA handling in nodes where there are no NA values.
        Default is 'weighted_random', which flips a weighted coin based on
        the number of samples going each direction -- e.g. if 70 points went
        left and 30 went right, would choose left for na 70% of the time.
        Option 'random' flips a 50-50 coin.

    random_state : int default=42
        A random seed that can be fixed for replicability purposes.

    References
    ----------

    Lucena, B. Exploiting Categorical Structure Using Tree-based Methods.
    Proceedings of the Twenty Third International Conference
    on Artificial Intelligence and Statistics, PMLR 108:2949-2958, 2020. 

    Lucena, B. StructureBoost: Efficient Gradient Boosting for Structured
    Categorical Variables. https://arxiv.org/abs/2007.04446
    """
    def __init__(self, num_trees, feature_configs,
                 mode='classification', subsample=1,
                 replace=True, max_depth=100, min_size_split=25,
                 min_leaf_size=None,
                 feat_sample_by_tree=1, feat_sample_by_node=1,
                 random_seed=0, na_unseen_action='weighted_random'):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.dec_tree_list = []
        self.feature_configs = feature_configs
        self._process_feature_configs()
        self.feat_list_full = list(self.feature_configs.keys())
        self.min_size_split = min_size_split
        self.max_depth = max_depth
        self.feat_sample_by_tree = feat_sample_by_tree
        self.feat_sample_by_node = feat_sample_by_node
        self.subsample = subsample
        self.replace = replace
        self.random_seed = random_seed
        self.na_unseen_action = na_unseen_action
        self.mode = mode
        if mode not in ['classification', 'regression']:
            warnings.warn('Mode not recognized')
        if min_leaf_size is None:
            if self.mode=='regression':
                self.min_leaf_size=5
            elif self.mode=='classification':
                self.min_leaf_size=1
        else:
            self.min_leaf_size = min_leaf_size


    def fit(self, X_train, y_train, verbose=True, interval=50):
        """Fits the model given training features and labels.

        Parameters
        ----------

        X_train : DataFrame
            A dataframe containing the features.  StructureBoost uses the
            column names rather than position to locate the features.  The
            set of features is determined by the feature_configs *not* 
            X_train.  Consequently, it is OK to pass in extra columns that
            are not used by the `feature_configs`.

        y_train : Series or numpy array
            The target values.  Must have length the same size as the number
            of rows in X_train.  Classification targets should be 0 or 1.
            Regression targets should be numerical.

        Attributes
        ----------

        dec_tree_list: A list of decision trees (represented as dicts).
        """
        # Initialize basic info
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if type(y_train) == pd.Series:
            y_train = y_train.to_numpy()
        if type(y_train) == np.ndarray:
            if self.mode == 'classification':
                y_train = y_train.astype(float)
            if self.mode == 'regression':
                y_train = y_train.astype(float)
        num_rows = X_train.shape[0]
        self.dec_tree_list = []
        col_list = list(X_train.columns)
        self.column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}

        # Initalize unique values dict
        self.unique_vals_dict = {}
        for feature in self.feature_configs.keys():
            if self.feature_configs[feature]['feature_type'] == 'numerical':
                self.unique_vals_dict[feature] = np.sort(
                                    pd.unique(X_train[feature].dropna()))

        if verbose:
            print('Beginning Tree Fitting')
        # Main loop to build trees
        for i in range(self.num_trees):
            if verbose and (i>0) and (i%interval==0):
                print('Trees fitted = {}'.format(i))
            # Sample the data to use for this tree
            rows_to_use = _get_rows_for_tree(num_rows, self.subsample,
                                             self.replace)

            # Determine which features to consider for this tree
            if self.feat_sample_by_tree < 1:
                feat_set_size = (self.feat_sample_by_tree *
                                 len(self.feat_list_full))
                feat_set_size = int(np.maximum(feat_set_size, 1))
                np.random.shuffle(self.feat_list_full)
                features_for_tree = self.feat_list_full[:feat_set_size]
            elif self.feat_sample_by_tree > 1:
                feat_set_size = int(self.feat_sample_by_tree)
                np.random.shuffle(self.feat_list_full)
                features_for_tree = self.feat_list_full[:feat_set_size]
            else:
                features_for_tree = self.feat_list_full

            X_train_to_use = X_train.iloc[rows_to_use, :]
            y_train_to_use = y_train[rows_to_use]

            # Add and train the next tree
            self.dec_tree_list.append(strfdt.StructureRFDecisionTree(
                            feature_configs=self.feature_configs,
                            feature_graphs=self.feature_graphs,
                            min_size_split=self.min_size_split,
                            max_depth=self.max_depth,
                            feat_sample_by_node=self.feat_sample_by_node,
                            min_leaf_size=self.min_leaf_size,
                            mode=self.mode))
            self.dec_tree_list[i].fit(X_train_to_use, y_train_to_use,
                                      feature_sublist=features_for_tree,
                                      uv_dict=self.unique_vals_dict)

        # Finally, randomize the direction in nodes having no NA
        if self.na_unseen_action == 'weighted_random':
            self.rerandomize_na_dir_weighted_all_trees()

        self.optimizable = np.all(np.array([dt.optimizable for dt in self.dec_tree_list]))
        if (self.optimizable):
            self.get_tensors_for_predict()

    def predict(self, X_test, int num_trees_to_use=-1, same_col_pos=True):
        """Returns a one-dimensional response - probability of 1 or numeric prediction

        If mode is classification (which is binary classification) then
        predict returns the probability of being in class '1'. (Note: unlike
        sklearn and other packages, `predict` will **not** return a 
        hard 1/0 prediction.  For `hard` prediction of class membership,
        threshold the probabilities appropriately.)

        If mode is regression, then predict returns the point estimate of
        the numerical target.

        Use `predict_proba` to get a 2-d matrix of class probabilities.

        `predict` uses a faster running variant when variable types are all numerical
        or categorical_int.  Use of categorical_str or graphical_voronoi variables
        force the use of a slower variant.

        Parameters
        ----------

        X_test : DataFrame or np.ndarray
            If you pass a numpy array, `predict` will expect the columns to
            be in the same positions as they were on the training dataframe.
            To see the current (expected) location of the columns, look at 
            the `column_to_int_dict` attribute of the model object.

            To remap the columns, use the `remap_predict_columns` method.

            If you pass a DataFrame, `predict` will look at the argument
            `same_col_pos` to determine whether to assume the column positions
            are the same as training or to actively find the columns.

        num_trees_to_use: int, default is -1
            By default, model will use the "num_trees_for_prediction"
            attribute to determine how many trees to use.  However,
            you can specify to use only the first k trees by setting
            num_trees_to_use to the value k.  This is useful for understanding
            model performance as a function of model_size.

        same_col_pos: int, default is True
            Used only when passing a DataFrame (not a numpy array) for X_test.  
            If True, then `predict` will assume the columns used are in the same 
            position as they were in the training DataFrame (which is slightly 
            faster).  If False, then `predict` will look at the column names and 
            remap them (which can be slower, but is sometimes more convenient). 
            This argument is ignored if X_test is a numpy array.

        Returns
        -------

        out_vec : array-like
            Returns a numpy array of length n, where n is the number of rows
            in X-test.
        """
        if (type(X_test)==np.ndarray):
            if self.optimizable:
                if self.optimized:
                    return(self._predict_fast(X_test, num_trees_to_use))
                else:
                    self.get_tensors_for_predict()
                    return(self._predict_fast(X_test, num_trees_to_use))
            else:
                # clunky.. make X_test into a pd.DataFrame with appr columns
                # but this would be an unusual situation
                inv_cti = {j:i for (i,j) in self.column_to_int_dict.items}
                col_list = [inv_cti[i] if i in inv_cti.keys() else '_X_'+str(i)
                            for i in X_test.shape[0]]
                return(self._predict_py(pd.DataFrame(X_test, columns=col_list), num_trees_to_use))

        elif (type(X_test)==pd.DataFrame):
            if same_col_pos:
                if self.optimizable:
                    if not self.optimized:
                        self.get_tensors_for_predict()
                    return(self._predict_fast(X_test.to_numpy(), num_trees_to_use))
                else:
                    return(self._predict_py(X_test, num_trees_to_use))
            else:
                return(self._predict_py(X_test, num_trees_to_use))

    def _predict_py(self, X_test, int num_trees_to_use=-1):
        cdef int i
        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        pred_vec = np.array([self.dec_tree_list[i].predict(X_test) for i in
                                range(num_trees_to_use)])
        return(np.mean(pred_vec, axis=0))

    def predict_proba(self, X_test, int num_trees_to_use=-1):
        """Returns a n x 2 matrix of class probabilities 

        Parameters
        ----------

        X_test : DataFrame
            A dataframe containing the features.  StructureBoost uses the
            column names rather than position to locate the features.  The
            set of features is determined by the feature_configs only.
            Consequently, it is OK to pass in extra columns to X_test that
            are not used by the particular model.

        num_trees_to_use: int, default is -1
            By default, model will use the "num_trees_for_prediction"
            attribute to determine how many trees to use.  However,
            you can specify to use only the first k trees by setting
            num_trees_to_use to the value k.  This is useful for understanding
            model performance as a function of model_size.

        Returns
        -------

        out_mat : array-like
            Returns a numpy array shape n x 2, where n is the number of rows
            in X-test.
        """
        pred_probs = self.predict(X_test, num_trees_to_use)
        return(np.vstack((1-pred_probs, pred_probs)).T)

    def _predict_fast(self, X_test, int num_trees_to_use=-1):
        cdef int i

        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        tree_pred_mat = predict_with_tensor_c(self.pred_tens_float[:num_trees_to_use,:,:],
                                              self.pred_tens_int[:num_trees_to_use,:,:],
                                              X_test)
        out_vec = np.mean(tree_pred_mat, axis=1)
        return(out_vec)

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

    def rerandomize_na_dir_weighted_all_trees(self):
        for i in range(len(self.dec_tree_list)):
            curr_tree = self.dec_tree_list[i].dec_tree
            if curr_tree:
                randomize_node_na_dir_weighted(curr_tree)

    def get_tensors_for_predict(self):
        if self.optimizable:
            cat_size = np.max(np.array([dt.get_max_split_size() for dt in self.dec_tree_list]))
            num_dt = len(self.dec_tree_list)
            max_nodes = np.max(np.array([dt.num_nodes for dt in self.dec_tree_list]))
            self.pred_tens_int = np.zeros((num_dt, max_nodes, cat_size+6), dtype=np.int64)-1
            self.pred_tens_float = np.zeros((num_dt, max_nodes, 2))
            for i in range(num_dt):
                self.convert_dt_to_matrix(i)
            self.optimized=True

        else:
            print("Model not optimizable for predict due to string or voronoi variable.")

    # # These are in dtm_float
    # cdef long THRESH = 0
    # cdef long NODE_VALUE = 1

    # # These are in dtm_int
    # cdef long NODE_TYPE = 0
    # cdef long FEATURE_COL = 1
    # cdef long LEFT_CHILD = 2
    # cdef long RIGHT_CHILD = 3
    # cdef long NA_LEFT = 4
    # cdef long NUM_CAT_VALS = 5
    # cdef long CAT_VALS_START = 6
    # # categorical values for left: 6 ... 6 + num_cat_vals-1

    # cdef long LEAF = 0
    # cdef long NUMER = 1
    # cdef long CATEG = 2


    def convert_dt_to_matrix(self, dt_num):
        curr_node = self.dec_tree_list[dt_num].dec_tree
        self.convert_subtree(curr_node, dt_num)

    def convert_subtree(self, node, dt_num):
        ni = node['node_index']
        if node['node_type']=='leaf':
            self.pred_tens_int[dt_num, ni, 0]= 0
            self.pred_tens_float[dt_num, ni, 1] = node['node_summary_val']
        else:
            if node['feature_type']=='numerical':
                self.pred_tens_float[dt_num, ni, 0] = node['split_val']
                self.pred_tens_int[dt_num, ni, 0]= 1
            elif node['feature_type']=='categorical_int':
                setlen = len(node['left_split'])
                self.pred_tens_int[dt_num, ni, 6:6+setlen] = np.fromiter(node['left_split'], int, setlen)
                self.pred_tens_int[dt_num, ni, 0]= 2
                self.pred_tens_int[dt_num, ni, 5]= setlen
            self.pred_tens_int[dt_num, ni, 1]=self.column_to_int_dict[node['split_feature']]
            self.pred_tens_int[dt_num, ni, 2]=node['left_child']['node_index']
            self.pred_tens_int[dt_num, ni, 3]=node['right_child']['node_index']
            self.pred_tens_int[dt_num, ni, 4]=node['na_left']
            self.convert_subtree(node['left_child'], dt_num)
            self.convert_subtree(node['right_child'], dt_num)


def _get_rows_for_tree(num_rows, subsample, replace):
    if (subsample == 1) and (not replace):
        return np.arange(num_rows)
    rows_to_return = int(np.ceil(num_rows*subsample))
    if replace:
        return np.random.randint(0, num_rows, rows_to_return)
    else:
        return np.random.choice(num_rows, rows_to_return, replace=False)




def randomize_node_na_dir_weighted(curr_node):
    if (('node_type' in curr_node.keys()) and 
                (curr_node['node_type'] == 'interior')):
        if ('na_dir_random' in curr_node.keys()) and (
             curr_node['na_dir_random'] == 1):
            lw = curr_node['left_child']['num_data_points']
            rw = curr_node['right_child']['num_data_points']
            curr_node['na_left'] = int(random.random() < (lw/(lw+rw)))
        if curr_node['left_child']['node_type'] == 'interior':
            randomize_node_na_dir_weighted(curr_node['left_child'])
        if curr_node['right_child']['node_type'] == 'interior':
            randomize_node_na_dir_weighted(curr_node['right_child'])

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_with_tensor_c(np.ndarray[double, ndim=3] dtm_float,
                      np.ndarray[long, ndim=3] dtm,
                      np.ndarray[double, ndim=2] feat_array):
    
    cdef long cat_vals_end
    cdef np.ndarray[double, ndim=2] res_mat = np.zeros((feat_array.shape[0], dtm.shape[0]))
    cdef long cn, ri, ind, j, k
    cdef double curr_val, ind_doub
    cdef bint at_leaf, found_val
    cdef np.ndarray[long, ndim=2] isnan_array = np.isnan(feat_array).astype(int)
    
    # These are in dtm_float
    cdef long THRESH = 0
    cdef long NODE_VALUE = 1

    # These are in dtm_int
    cdef long NODE_TYPE = 0
    cdef long FEATURE_COL = 1
    cdef long LEFT_CHILD = 2
    cdef long RIGHT_CHILD = 3
    cdef long NA_LEFT = 4
    cdef long NUM_CAT_VALS = 5
    cdef long CAT_VALS_START = 6
    # categorical values for left: 6 ... whatever

    cdef long LEAF = 0
    cdef long NUMER = 1
    cdef long CATEG = 2

    for k in range(dtm.shape[0]):
        for ri in range(feat_array.shape[0]):
            cn = 0
            at_leaf = 0
            while not at_leaf:
                cn = int(cn)
                if dtm[k,cn, NODE_TYPE]==LEAF:
                    at_leaf = 1
                    res_mat[ri,k] = dtm_float[k,cn, NODE_VALUE]
                elif dtm[k,cn, NODE_TYPE]==NUMER:
                    ind = dtm[k,cn, FEATURE_COL]
                    if isnan_array[ri,ind]:
                        cn = dtm[k,cn, LEFT_CHILD] if dtm[k,cn, NA_LEFT] else dtm[k,cn, RIGHT_CHILD]
                    else:
                        curr_val = feat_array[ri,ind]
                        cn = dtm[k,cn, LEFT_CHILD] if curr_val<dtm_float[k,cn, THRESH] else dtm[k,cn, RIGHT_CHILD]
                elif dtm[k,cn, NODE_TYPE]==CATEG:
                    curr_val = feat_array[ri,dtm[k,cn, FEATURE_COL]]
                    found_val = 0
                    j = CAT_VALS_START
                    cat_vals_end = CAT_VALS_START + dtm[k, cn, NUM_CAT_VALS]
                    while ((not found_val) & (j<cat_vals_end)):
                        if curr_val==dtm[k,cn, j]:
                            found_val=1
                        else:
                            j+=1
                    cn = dtm[k,cn, LEFT_CHILD] if found_val else dtm[k,cn, RIGHT_CHILD]
    return(res_mat)
