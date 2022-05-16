# cython: profile=True
# cython: language_level=3

"""Structured Gradient Boosting using graphs"""
import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
from libc.math cimport log as clog
from libc.math cimport exp
cimport numpy as np
cimport cython


class StructureBoost(object):
    """Gradient Boosting model allowing categorical structure.

    This is the appropriate model to use for regression and/or binary
    classification.  For multiclass problems use `StructureBoostMulti`.

    Requires a feature-specific configuration -- see docs and examples:
        www.github.com/numeristical/structureboost/examples

    Categorical variables require a graph as part of their feature config.

    Uses Newton steps based on first and second derivatives of loss fn.
    Loss function defaults to entropy (log_loss) for classification and
    mean squared error for regression.  A specific loss function can be used
    by providing a loss_fn with first and second derivatives.

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

    learning_rate: float, default is .02
        The "step size" to take when adding each new tree to the model.
        We recommend using a relatively low learning rate combined with
        early stopping.

    max_depth : int, default is 3
        The maximum depth to use when building trees. This is an important
        parameter.  Smaller values are more likely to underfit and 
        larger values more likely to overfit.

    loss_fn : 'entropy', 'mse', or tuple of functions
        Loss fn to use.  Default depends on 'mode' - uses 'entropy' for
        classification and 'mse' (mean squared error) for regression.
        If you wish, you can pass in a tuple of functions for the first
        and second derivatives of your loss function.

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

    gamma : numeric, default is 0
        Regularization parameter that specifies the minimum "gain"
        required to execute a split (as in XGBoost). Larger values
        indicate more regularization.

    reg_lambda : numeric, default is 1
        Regularization parameter that acts as an L2 penalty on
        coefficient size (as in XGBoost).  Larger values indicate
        more regularization.

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

    Lucena, B. "Exploiting Categorical Structure with Tree-Based Methods."
    Proceedings of the Twenty Third International Conference on Artificial
    Intelligence and Statistics, PMLR 108:2949-2958, 2020.

    Lucena, B. StructureBoost: Efficient Gradient Boosting for Structured
    Categorical Variables. https://arxiv.org/abs/2007.04446
    """
    def __init__(self, num_trees, feature_configs,
                 mode='classification',
                 loss_fn=None, subsample=1,
                 initial_model=None,
                 replace=True, min_size_split=25, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.02,
                 random_seed=0, na_unseen_action='weighted_random'):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.feature_configs = feature_configs
        self._process_feature_configs()
        self.feat_list_full = list(self.feature_configs.keys())
        self.min_size_split = min_size_split
        self.max_depth = max_depth
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.feat_sample_by_tree = feat_sample_by_tree
        self.feat_sample_by_node = feat_sample_by_node
        self.loss_fn = loss_fn
        self.subsample = subsample
        self.initial_model = initial_model
        self.replace = replace
        self.random_seed = random_seed
        self.na_unseen_action = na_unseen_action
        self.num_classes=2
        self.optimized = False
        self.mode = mode
        if mode not in ['classification', 'regression']:
            warnings.warn('Mode not recognized')
        if loss_fn is None:
            if mode == 'classification':
                loss_fn = 'entropy'
            else:
                loss_fn = 'mse'
        if type(loss_fn) == tuple:
            self.loss_fn_der_1 = loss_fn[0]
            self.loss_fn_der_2 = loss_fn[1]
        elif (loss_fn == 'entropy'):
            self.loss_fn_der_1 = c_entropy_link_der_1
            self.loss_fn_der_2 = c_entropy_link_der_2
        elif loss_fn == 'mse':
            self.loss_fn_der_1 = _mse_der_1
            self.loss_fn_der_2 = _mse_der_2

        self._validate_feature_config()

    def _validate_feature_config(self):
        valid_ftypes = ['numerical', 'categorical_int',
                        'categorical_str', 'graphical_voronoi']
        cat_smtypes = ['span_tree', 'contraction','onehot']
        for fn in self.feature_configs.keys():
            if 'feature_type' not in self.feature_configs[fn]:
                w_str = '"feature_type" not configured in feature {}'.format(fn)
                warnings.warn(w_str)
            elif self.feature_configs[fn]['feature_type'] not in valid_ftypes:
                w_str = 'Unknown feature_type in feature {}'.format(fn)
                warnings.warn(w_str)
            elif self.feature_configs[fn]['feature_type']=='numerical':
                if 'max_splits_to_search' not in self.feature_configs[fn]:
                    w_str = '"max_splits_to_search" not configured in feature {}'.format(fn)
                    warnings.warn(w_str)

            elif self.feature_configs[fn]['feature_type'] in ['categorical_int',
                                                            'categorical_str']:
                # Check for existence and connectivity of graphs
                if ('graph' not in self.feature_configs[fn]):
                    if (self.feature_configs[fn]['split_method'] != 'onehot'):
                        w_str = 'graph not configured in feature {}'.format(fn)
                        warnings.warn(w_str)
                elif not graphs.is_connected(self.feature_configs[fn]['graph']):
                    w_str = 'graph not connected in feature {}'.format(fn)
                    warnings.warn(w_str)
                
                # Check for proper parameters existence
                if 'split_method' not in self.feature_configs[fn]:
                    w_str='"split_method" not specified in feature {}'.format(fn)
                    warnings.warn(w_str)
                elif self.feature_configs[fn]['split_method'] not in cat_smtypes:
                    w_str = 'Unknown split_method in feature {}'.format(fn)
                    warnings.warn(w_str)
                elif self.feature_configs[fn]['split_method'] == 'span_tree':
                    if 'num_span_trees' not in self.feature_configs[fn]:
                        w_str = '"num_span_trees" not configured in feature {}'.format(fn)
                        warnings.warn(w_str)
                elif self.feature_configs[fn]['split_method'] == 'contraction':
                    if 'contraction_size' not in self.feature_configs[fn]:
                        w_str = '"contraction_size" not configured in feature {}'.format(fn)
                        warnings.warn(w_str)
                    if 'max_splits_to_search' not in self.feature_configs[fn]:
                        w_str = '"max_splits_to_search" not configured in feature {}'.format(fn)
                        warnings.warn(w_str)

    def _process_y_data(self, y_data):
        if type(y_data) == pd.Series:
            y_data = y_data.to_numpy().astype(float)
        elif type(y_data) == np.ndarray:
            y_data = y_data.astype(float)
        return(y_data)

    def _get_initial_pred(self, X_train, y_train):
        if self.initial_model is not None:
            if (self.mode == 'classification'):
                start_probs = self.initial_model.predict_proba(X_train)[:,1]
                start_probs = np.clip(start_probs, 1e-15, 1-1e-15)
                return(np.log(start_probs / (1-start_probs)))
            else:
                return(self.initial_model.predict(X_train))
        else:
            if (self.mode == 'classification'):
                prob_est = np.mean(y_train)
                return((np.log(prob_est / 
                                    (1-prob_est)))*np.ones(X_train.shape[0]))
            else:
                return(np.mean(y_train)*np.ones(X_train.shape[0]))

    def _add_train_next_tree(self, features_for_tree, X_train, y_g_h_train, index):
        self.dec_tree_list.append(stdt.StructureDecisionTree(
                        feature_configs=self.feature_configs,
                        feature_graphs=self.feature_graphs,
                        min_size_split=self.min_size_split,
                        gamma=self.gamma, max_depth=self.max_depth,
                        reg_lambda=self.reg_lambda,
                        feat_sample_by_node=self.feat_sample_by_node))
        self.dec_tree_list[index].fit(X_train, y_g_h_train,
                                  feature_sublist=features_for_tree,
                                  uv_dict=self.unique_vals_dict)

    def process_curr_answer(self, curr_answer):
        return curr_answer

    def compute_gh_mat(self, y_train, curr_answer):
        y_g_vec = self.loss_fn_der_1(y_train, curr_answer)
        y_h_vec = self.loss_fn_der_2(y_train, curr_answer)
        y_g_h_mat = np.vstack((y_g_vec, y_h_vec)).T
        return y_g_h_mat


    def _compute_loss(self, y_true, pred):
        if ((self.mode == 'classification')):
            return(my_log_loss(y_true, 1/(1+np.exp(-pred))))
        else:
            return(my_mean_squared_error(y_true, pred))


    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
            early_stop_past_steps=0, choose_best_eval=True):
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

        eval_set : tuple or list
            A tuple or list contaning the X values as the first element and
            y values of the second element.  Will output the model's loss
            function at each interval as specified by eval_freq.

        eval_freq : int, default is 10
            the number of steps between successive outputs of the
            loss fn value of the current model.  Each set of eval_freq number
            of trees is considered a "step" for `early_stop_past_steps`

        early_stop_past_steps : int, default is 0
            How far back to look to determine early stopping. Default is 0,
            which turns off early_stopping. If set at m, it will
            compare current loss to loss m "steps" ago to decide whether or
            not to stop. (Note that a "step" here means `eval_freq` number
            of trees). Larger numbers give more room for allowing temporary
            increases in the loss before giving up and stopping.

        choose_best_eval : bool, default is True
            When using early stopping, this determines whether or not to
            revisit the previous "step" evaluations and choose the best model.


        Attributes
        ----------

        dec_tree_list: A list of decision trees (represented as dicts).

        num_trees_for_prediction: The number of trees to use for prediction
            (by default).  Used to truncate the model rather than deleting
            trees.
        """
        # Initialize random seeds
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.dec_tree_list = []

        y_train = self._process_y_data(y_train)
        self.initial_pred = self._get_initial_pred(X_train, y_train)

        num_rows = X_train.shape[0]
        col_list = list(X_train.columns)
        self.column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        curr_answer = self.initial_pred

        # Initalize unique values dict
        self.unique_vals_dict = {}
        for feature in self.feature_configs.keys():
            if self.feature_configs[feature]['feature_type'] == 'numerical':
                self.unique_vals_dict[feature] = np.sort(
                                    pd.unique(X_train[feature].dropna()))

        # Initalize eval_set related
        self.eval_results = np.zeros(np.floor(self.num_trees/eval_freq).astype(int))
        if eval_set is not None:
            X_valid = eval_set[0]
            y_valid = eval_set[1]
            y_valid = self._process_y_data(y_valid)
            # below, we use X_valid and y_train *not a mistake* 
            # if there is initial model, we apply it to X_valid
            # otherwise we use the marginal averages from y_train
            curr_valid_answer = self._get_initial_pred(X_valid, y_train)
            curr_valid_loss = self._compute_loss(y_valid, curr_valid_answer)
            print("i={}, eval_set_loss = {}".format(0, curr_valid_loss))

        # Main loop to build trees
        stop_now = False
        for i in range(self.num_trees):

            # Get predictions of current model
            if (i > 0):
                curr_answer = (curr_answer + self.learning_rate *
                               self.dec_tree_list[i-1].predict(X_train))

                # handle eval_set / early_stopping related tasks
                if eval_set is not None:
                    curr_valid_answer = (curr_valid_answer +
                                         self.learning_rate *
                                         self.dec_tree_list[i-1].predict(
                                            X_valid))
                    if ((i+1) % eval_freq == 1):
                        curr_loss = self._compute_loss(y_valid, curr_valid_answer)
                        print("i={}, eval_set_loss = {}".format(i, curr_loss))
                        curr_step = np.floor((i+1) /
                                             eval_freq).astype(int)-1
                        self.eval_results[curr_step] = curr_loss
                        if curr_step > early_stop_past_steps:
                            compare_loss = np.min(self.eval_results[:(
                                           curr_step-early_stop_past_steps+1)])
                            if (curr_loss > compare_loss):
                                stop_now = True
                                print("""Stopping early: curr_loss of {}
                                        exceeds compare_loss of {}"""
                                      .format(curr_loss, compare_loss))
                if stop_now:
                    if choose_best_eval:
                        self.num_trees_for_prediction = ((
                            np.argmin(self.eval_results[:curr_step+1])+1) *
                            eval_freq)
                    break

            # Get first and second derivatives of loss fn
            # relative to current prediction
            curr_answer = self.process_curr_answer(curr_answer)
            y_g_h_mat = self.compute_gh_mat(y_train, curr_answer)

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
            y_g_h_to_use = y_g_h_mat[rows_to_use, :]

            # Add and train the next tree
            self._add_train_next_tree(features_for_tree, X_train_to_use, y_g_h_to_use, i)
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
        if self.initial_model is None:
            # todo: revisit below
            out_vec = self.initial_pred[0]*np.ones(X_test.shape[0])
        elif self.mode=='classification':
            out_vec = self.initial_model.predict_proba(X_test)[:,1]
            out_vec = np.clip(out_vec, 1e-15, 1-1e-15)
            out_vec = np.log(out_vec/(1-out_vec))
        else:
            out_vec = self.initial_model.predict(X_test)
        for i in range(num_trees_to_use):
            out_vec = (out_vec + self.learning_rate *
                       self.dec_tree_list[i].predict(X_test))
        if self.mode == 'classification':
            return(1/(1+np.exp(-out_vec)))
        else:
            return(out_vec)

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
        if self.initial_model is None:
            # todo: revisit below
            out_vec = self.initial_pred[0]*np.ones(X_test.shape[0])
        elif self.mode=='classification':
            out_vec = self.initial_model.predict_proba(X_test)[:,1]
            out_vec = np.clip(out_vec, 1e-15, 1-1e-15)
            out_vec = np.log(out_vec/(1-out_vec))
        else:
            out_vec = self.initial_model.predict(X_test)
        tree_pred_mat = predict_with_tensor_c(self.pred_tens_float[:num_trees_to_use,:,:],
                                              self.pred_tens_int[:num_trees_to_use,:,:],
                                              X_test)
        out_vec = out_vec + self.learning_rate*np.sum(tree_pred_mat, axis=1)
        if self.mode == 'classification':
            return(1/(1+np.exp(-out_vec)))
        else:
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

    def remap_predict_columns(self, new_col_struct):
        """Indicate which column in numpy array each feature is located.

        When predicting on a numpy array (as opposed to a pandas DataFrame),
        it is not explicit which column of the numpy array corresponds to 
        which feature.  By default, the predict method will assume that the
        columns are in the same location they were for the training data.  If
        this is not the case, the user can 'remap' the columns using this method.

        Parameters
        ----------

        new_col_struct : list, dict, pd.DataFrame, or pd.core.indexes.base.Index

        Data structure to show which features map to which columns.  

        If a list (or
        a pandas index) then the order of the strings in the list indicate the 
        location in the numpy array (it is permissible to have extra columns that
        are not used by the model)

        If a dict, the keys should be the feature names and the corresponding values
        should be the integers indicating the column index.

        If a DataFrame, it will use the columns attribute to map features to indices.
        This is useful if you want to then call `predict` on df.to_numpy()
        """
        if type(new_col_struct)==pd.DataFrame:
            col_list = new_col_struct.columns
            self.column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
            self.get_tensors_for_predict()
        elif type(new_col_struct)==dict:
            self.column_to_int_dict = new_col_struct
            self.get_tensors_for_predict()
        elif ((type(new_col_struct)==list) or 
                    (type(new_col_struct)==pd.core.indexes.base.Index)):
            self.column_to_int_dict = {new_col_struct[i]: i for i in 
                                                        range(len(new_col_struct))}
            self.get_tensors_for_predict()
        else:
            w_str='new_col_struct not of appropriate type - mapping not changed'
            warnings.warn(w_str)

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


def _mse_der_1(y_true, y_pred, eps=1e-15):
    return(2*(y_pred-y_true))


def _mse_der_2(y_true, y_pred, eps=1e-15):
    return(2*np.ones(len(y_pred)))


def _entropy_link_der_1(y_true, z_pred, eps=1e-15):
    return(-y_true*(1/(1+np.exp(z_pred))) + (1-y_true) *
           (1/(1+np.exp(-z_pred))))


def _entropy_link_der_2(y_true, z_pred, eps=1e-15):
    z_pred_exp = np.exp(z_pred)
    minus_z_pred_exp = np.exp(-z_pred)
    denom_1 = (1+z_pred_exp)*(1+z_pred_exp)
    denom_2 = (1+minus_z_pred_exp)*(1+minus_z_pred_exp)
    return(y_true*(z_pred_exp/denom_1) +
           (1-y_true) * (minus_z_pred_exp/denom_2))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def c_entropy_link_der_1(np.ndarray[double] y_true,
                         np.ndarray[double] z_pred):
    cdef int N = y_true.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef double denom1, denom2
    cdef long i

    for i in range(N):
        denom_1 = 1+exp(z_pred[i])
        denom_2 = 1+exp(-z_pred[i])
        Y[i] = (-y_true[i]*(1/denom_1) + (1-y_true[i]) * (1/denom_2))
    return Y


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def c_entropy_link_der_2(np.ndarray[double] y_true,
                         np.ndarray[double] z_pred):
    cdef int N = y_true.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef double z_pred_exp, minus_z_pred_exp, denom1, denom2
    cdef long i

    for i in range(N):
        z_pred_exp = exp(z_pred[i])
        minus_z_pred_exp = exp(-z_pred[i])
        denom_1 = (1+z_pred_exp)*(1+z_pred_exp)
        denom_2 = (1+minus_z_pred_exp)*(1+minus_z_pred_exp)
        Y[i] = (y_true[i]*(z_pred_exp/denom_1) + (1-y_true[i]) *
                          (minus_z_pred_exp/denom_2))
    return Y


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
                
        

ctypedef np.int64_t dtype_int64_t 

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_entropy_link_der_1_vec(np.ndarray[dtype_int64_t] y_true,
#                          double[:,:] phi_pred):
#     cdef int N = y_true.shape[0]
#     cdef int m = phi_pred.shape[1]
#     cdef double[:,:] Y = np.zeros((N,m))
#     #Y = np.zeros((N, m), dtype=np.dtype('float64'))
#     cdef float phi_exp_sum = 0
    
#     for i in range(N):
#         for j in range(m):
#             phi_exp_sum = 0
#             for k in range(m):
#                 phi_exp_sum+=exp(phi_pred[i,k])
#             if j==y_true[i]:
#                 Y[i,j]=(exp(phi_pred[i,j])/phi_exp_sum)-1
#             else:
#                 Y[i,j]=(exp(phi_pred[i,j])/phi_exp_sum)
                
#     return np.asarray(Y)


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_entropy_link_der_2_vec(np.ndarray[dtype_int64_t] y_true,
#                          double[:,:] phi_pred):
#     cdef int N = y_true.shape[0]
#     cdef int m = phi_pred.shape[1]
#     cdef double[:,:] Y = np.zeros((N,m))
#     #Y = np.zeros((N, m), dtype=np.dtype('float64'))
#     cdef float phi_exp_sum = 0
    
#     for i in range(N):
#         for j in range(m):
#             phi_exp_sum = 0
#             for k in range(m):
#                 phi_exp_sum+=exp(phi_pred[i,k])
#             Y[i,j]=-(exp(phi_pred[i,j])*(exp(phi_pred[i,j]-phi_exp_sum)/
#                                              (phi_exp_sum*phi_exp_sum)))                
#     return np.asarray(Y)

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_str_entropy_link_der_1_vec(np.ndarray[dtype_int64_t] y_true,
#                                  double[:,:] phi_pred, 
#                                  double[:] weight_vec, 
#                                  dtype_int64_t[:,:,:] rp_tensor, 
#                                  int num_part):
#     cdef int N = y_true.shape[0]
#     cdef int m = phi_pred.shape[1]
#     cdef int ind = 0
#     cdef int qqq, xyz
#     cdef double curr_wt
#     cdef dict md
#     cdef double[:,:] Y = np.zeros((N,m))
#     #Y = np.zeros((N, m), dtype=np.dtype('float64'))
#     cdef double all_sum = 0
#     cdef double set_sum = 0

#     for qqq in range(num_part):
#         curr_wt = weight_vec[ind]
        
#         for i in range(N):
#             # get sum of exp phi for this row
#             all_sum = 0
#             for k in range(m):
#                 all_sum+=exp(phi_pred[i,k])
#             xyz=0
#             while (rp_tensor[qqq,xyz,y_true[i]]==0):
#                 xyz+=1
#             set_sum = 0
#             for t in range(m):
#                 if (rp_tensor[qqq,xyz,t]==1):
#                     set_sum+=exp(phi_pred[i,t])
#             for j in range(m):

#                 # if j in set containing y_true[i]
#                 if (rp_tensor[qqq,xyz,j]==1):
#                     #Y[i,j]+= value when j in T * weight
#                     Y[i,j]+=(-curr_wt * (exp(phi_pred[i,j])*(all_sum-set_sum)/
#                                         (set_sum*all_sum)))
#                 else:
#                     #Y[i,j]+= value when j not in T * weight
#                     Y[i,j]+=curr_wt * exp(phi_pred[i,j])/all_sum
                
#     return np.asarray(Y)

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_str_entropy_link_der_2_vec(np.ndarray[dtype_int64_t] y_true,
#                                  double[:,:] phi_pred, 
#                                  double[:] weight_vec, 
#                                  dtype_int64_t[:,:,:] rp_tensor, 
#                                  int num_part):
#     cdef int N = y_true.shape[0]
#     cdef int m = phi_pred.shape[1]
#     cdef int ind = 0
#     cdef int qqq, xyz
#     cdef double curr_wt
#     cdef dict md
#     cdef double[:,:] Y = np.zeros((N,m))
#     #Y = np.zeros((N, m), dtype=np.dtype('float64'))
#     cdef double all_sum = 0
#     cdef double set_sum = 0

#     for qqq in range(num_part):
#         curr_wt = weight_vec[ind]
        
#         for i in range(N):
#             # get sum of exp phi for this row
#             all_sum = 0
#             for k in range(m):
#                 all_sum+=exp(phi_pred[i,k])
#             xyz=0
#             while (rp_tensor[qqq,xyz,y_true[i]]==0):
#                 xyz+=1
#             set_sum = 0
#             for t in range(m):
#                 if (rp_tensor[qqq,xyz,t]==1):
#                     set_sum+=exp(phi_pred[i,t])
#             for j in range(m):

#                 # if j in set containing y_true[i]
#                 jt = (exp(phi_pred[i,j])/set_sum)
#                 jk = (exp(phi_pred[i,j])/all_sum)
#                 if (rp_tensor[qqq,xyz,j]==1):
#                     #Y[i,j]+= value when j in T * weight
#                     Y[i,j]+=(-curr_wt * (jt-jk+jk*jk-jt*jt))
#                 else:
#                     #Y[i,j]+= value when j not in T * weight
#                     Y[i,j]+=(-curr_wt * (-jk+jk*jk))
                
#     return np.asarray(Y)


def my_log_loss(y_true, y_pred, eps=1e-16):
    y_pred = np.clip(y_pred, eps, (1-eps))
    out_val = -np.mean(y_true*(np.log(y_pred)) + (1-y_true)*np.log(1-y_pred))
    return out_val


def my_log_loss_vec(y_true_mat, y_pred, eps=1e-16):
    y_pred = np.clip(y_pred, eps, (1-eps))
    out_val = -np.mean(np.sum(y_true_mat*np.log(y_pred),axis=1))
    return out_val


def my_mean_squared_error(y_true, y_pred):
    return(np.mean((y_true-y_pred)*(y_true-y_pred)))


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
