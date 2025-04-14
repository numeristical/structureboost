# cython: profile=True
# cython: language_level=3

"""Structured Gradient Boosting using graphs"""
import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
# from utils import get_basic_config, default_config_dict
# Always run into issues trying to import from utils
# I just added the fns to structure_gb ...

from libc.math cimport log as clog
from libc.math cimport exp
cimport numpy as np
np.import_array()
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

    prec_digits : int default = 6
        Number of digits to use for finding unique values in floats. This
        is a technical point where two floats may be treated differently
        even though they are the same value if this setting is too high.


    References
    ----------

    Lucena, B. "Exploiting Categorical Structure with Tree-Based Methods."
    Proceedings of the Twenty Third International Conference on Artificial
    Intelligence and Statistics, PMLR 108:2949-2958, 2020.

    Lucena, B. StructureBoost: Efficient Gradient Boosting for Structured
    Categorical Variables. https://arxiv.org/abs/2007.04446
    """
    def __init__(self, num_trees, feature_configs=None,
                 mode='classification',
                 loss_fn=None, subsample=1,
                 initial_model=None,
                 replace=True, min_size_split=25, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.02,
                 random_seed=0, na_unseen_action='weighted_random',
                 prec_digits=6, default_configs=None):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        if feature_configs is not None:
            self.feature_configs = feature_configs.copy()
            self._process_feature_configs()
            self.feat_list_full = list(self.feature_configs.keys())
        else:
            self.feature_configs=None
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
        self.prec_digits=prec_digits
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

        if self.feature_configs is not None:
            self._validate_feature_config()
        if default_configs is None:
            self.default_configs = default_config_dict()

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

    def compute_gh_mat(self, y_train, curr_answer, index):
        y_g_vec = self.loss_fn_der_1(y_train, curr_answer)
        y_h_vec = self.loss_fn_der_2(y_train, curr_answer)
        y_g_h_mat = np.vstack((y_g_vec, y_h_vec)).T
        return y_g_h_mat


    def _compute_loss(self, y_true, pred):
        if ((self.mode == 'classification')):
            return(my_log_loss(y_true, 1/(1+np.exp(-pred))))
        else:
            return(my_mean_squared_error(y_true, pred))

    def _update_curr_answer(self, curr_answer, X_train, index):
        if (index>0):
            curr_answer = (curr_answer + self.learning_rate *
                                    self.dec_tree_list[index-1].predict(X_train))
        return(curr_answer)

    def _get_rows_for_tree(self, num_rows):
        if (self.subsample == 1) and (not self.replace):
            return np.arange(num_rows)
        rows_to_return = int(np.ceil(num_rows*self.subsample))
        if self.replace:
            return np.random.randint(0, num_rows, rows_to_return)
        else:
            return np.random.choice(num_rows, rows_to_return, replace=False)


    def _get_features_for_tree(self):
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

    def _output_loss_check_stop(self, y_valid, curr_valid_answer, i):
        stop_now=False
        if ((i+1) % self.eval_freq == 1) and (i>0):
            curr_loss = self._compute_loss(y_valid, curr_valid_answer)
            if self.verbose:
                print("i={}, eval_set_loss = {}".format(i, curr_loss))
            self.curr_step = np.floor((i+1) /
                                 self.eval_freq).astype(int)-1
            self.eval_results[self.curr_step] = curr_loss
            if ((self.early_stop_past_steps>=0) and 
                    (self.curr_step > self.early_stop_past_steps)):
                self.best_step = np.argmin(self.eval_results[:(self.curr_step+1)])
                if ((self.curr_step-self.best_step) >= self.early_stop_past_steps):
                    stop_now = True
                    if self.verbose:
                        print("""Stopping early: low pt was {} steps ago"""
                            .format((self.curr_step-self.best_step)))
                    if self.choose_best_eval:
                        self.num_trees_for_prediction = ((
                            np.argmin(self.eval_results[:self.curr_step+1])+1) *
                            self.eval_freq)
        return(stop_now)

    def _initialize_uv_dict(self, X_train):
        self.unique_vals_dict = {}
        for feature in self.feature_configs.keys():
            if self.feature_configs[feature]['feature_type'] == 'numerical':
                self.unique_vals_dict[feature] = np.sort(
                                    pd.unique(np.round(X_train[feature].dropna(), 
                                        decimals=self.prec_digits)))

    def _init_eval_set_info(self, eval_set, y_train):
        self.eval_results = np.zeros(np.floor(
                                self.num_trees/self.eval_freq).astype(int))
        X_valid = eval_set[0]
        y_valid = eval_set[1]
        y_valid = self._process_y_data(y_valid)
        # below, we use X_valid and y_train *not a mistake* 
        # if there is initial model, we apply it to X_valid
        # otherwise we use the marginal averages from y_train
        curr_valid_answer = self._get_initial_pred(X_valid, y_train)
        curr_valid_loss = self._compute_loss(y_valid, curr_valid_answer)
        if self.verbose:
            print("i={}, eval_set_loss = {}".format(0, curr_valid_loss))
        return(X_valid, y_valid, curr_valid_answer)

    def _output_no_evalset(self, index):
        if (((index+1) % self.eval_freq == 1) and self.verbose):
                print("i={}".format(index))


    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
            early_stop_past_steps=0, choose_best_eval=True, verbose=1):
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
            which means it will stop immediately if the loss increases on the
            eval set. If set at m, it will stop if it has been m steps since
            the last "new low" was achieved. If set to -1, will deactivate
            early stopping. (Note that a "step" here means `eval_freq` number
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
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if self.feature_configs is None:
            self.feature_configs = get_basic_config(X_train,self.default_configs)
            self._process_feature_configs()
            self.feat_list_full = list(self.feature_configs.keys())
        self.dec_tree_list = []
        self.verbose = verbose
        self.early_stop_past_steps = early_stop_past_steps
        self.eval_freq = eval_freq
        self.choose_best_eval = choose_best_eval
        self.has_eval_set = (eval_set is not None)
        num_rows = X_train.shape[0]
        col_list = list(X_train.columns)
        self.column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        self._initialize_uv_dict(X_train)

        y_train = self._process_y_data(y_train)
        self.initial_pred = self._get_initial_pred(X_train, y_train)
        curr_answer = self.initial_pred
        if self.has_eval_set:
            X_valid, y_valid, curr_valid_answer = self._init_eval_set_info(
                                                            eval_set, y_train)

        # Main loop to build trees
        stop_now = False
        for i in range(self.num_trees):
            curr_answer = self._update_curr_answer(curr_answer, X_train, i)
            # handle eval_set / early_stopping related tasks
            if self.has_eval_set:
                curr_valid_answer = self._update_curr_answer(curr_valid_answer,
                                                    X_valid, i)
                stop_now = self._output_loss_check_stop(y_valid,
                                    curr_valid_answer, i)
                if stop_now:
                    break
            else:
                self._output_no_evalset(i)

            # Get first and second derivatives of loss fn for current prediction
            curr_answer = self.process_curr_answer(curr_answer)
            y_g_h_mat = self.compute_gh_mat(y_train, curr_answer, i)

            # Determine rows and features to use for this tree
            rows_to_use = self._get_rows_for_tree(num_rows)
            features_for_tree = self._get_features_for_tree()
            X_train_to_use = X_train.iloc[rows_to_use, :]
            y_g_h_to_use = y_g_h_mat[rows_to_use, :]

            # Add and train the next tree
            self._add_train_next_tree(features_for_tree, X_train_to_use, y_g_h_to_use, i)

        # Post-processing steps
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

    def predict_shap(self, X_test, int num_trees_to_use=-1, same_col_pos=True):
        if (type(X_test)==np.ndarray):
            if self.optimizable:
                if self.optimized:
                    return(self._predict_shap(X_test, num_trees_to_use))
                else:
                    self.get_tensors_for_predict()
                    return(self._predict_shap(X_test, num_trees_to_use))
            else:
                print("Shap values not available for categorical string or voronoi vars")
        elif (type(X_test)==pd.DataFrame):
            if same_col_pos:
                if self.optimizable:
                    if not self.optimized:
                        self.get_tensors_for_predict()
                    return(self._predict_shap(X_test.to_numpy(), num_trees_to_use))
                else:
                    print("Shap values not available for categorical string or voronoi vars")
            else:
                print('columns must be in correct order for shap values')

    def _predict_shap(self, 
                    np.ndarray[double, ndim=2] X_test, int num_trees_to_use=-1):
        cdef long i,j
        cdef long nrows = X_test.shape[0]
        cdef long nfeat = X_test.shape[1] #not including intercept
        cdef double start_pred

        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        phi_out = np.zeros((nrows, nfeat+1))
        start_pred=0
        if self.initial_model is None:
            start_pred = self.initial_pred[0]
        for j in range(nrows):
            phi_out[j,nfeat] += start_pred
        for i in range(num_trees_to_use):
            dt_mat_int = self.pred_tens_int[i,:,:]
            dt_mat_float = self.pred_tens_float[i,:,:]
            for j in range(nrows):
                curr_pt = X_test[j,:]
                phi_out[j,:]+= tree_shap_single_pt(dt_mat_int, dt_mat_float,
                    self.max_depth, curr_pt) * self.learning_rate
        return(phi_out)


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
            self.pred_tens_int = np.zeros((num_dt, max_nodes, cat_size+6), dtype=np.int32)-1
            self.pred_tens_float = np.zeros((num_dt, max_nodes, 3))
            for i in range(num_dt):
                self.convert_dt_to_matrix(i)
            self.optimized=True
        else:
            print("Model not optimizable for predict due to string or voronoi variable.")

    def convert_dt_to_matrix(self, dt_num):
        curr_node = self.dec_tree_list[dt_num].dec_tree
        self.convert_subtree(curr_node, dt_num)

    def convert_subtree(self, node, dt_num):
        ni = node['node_index']
        if node['node_type']=='leaf':
            self.pred_tens_int[dt_num, ni, 0]= 0
            self.pred_tens_float[dt_num, ni, 1] = float(node['num_data_points'])
            self.pred_tens_float[dt_num, ni, 2] = node['node_summary_val']
        else:
            self.pred_tens_float[dt_num, ni, 1] = float(node['num_data_points'])
            if node['feature_type']=='numerical':
                self.pred_tens_float[dt_num, ni, 0] = node['split_val']
                self.pred_tens_int[dt_num, ni, 0]= 1
            elif node['feature_type']=='categorical_int':
                setlen = len(node['left_split'])
                self.pred_tens_int[dt_num, ni, 6:6+setlen] = np.fromiter(node['left_split'], np.int32, setlen)
                self.pred_tens_int[dt_num, ni, 0]= 2
                self.pred_tens_int[dt_num, ni, 5]= setlen
            self.pred_tens_int[dt_num, ni, 1]=self.column_to_int_dict[node['split_feature']]
            self.pred_tens_int[dt_num, ni, 2]=node['left_child']['node_index']
            self.pred_tens_int[dt_num, ni, 3]=node['right_child']['node_index']
            self.pred_tens_int[dt_num, ni, 4]=node['na_left']
            self.convert_subtree(node['left_child'], dt_num)
            self.convert_subtree(node['right_child'], dt_num)


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
                      np.ndarray[np.int32_t, ndim=3] dtm,
                      np.ndarray[double, ndim=2] feat_array):
    
    cdef long cat_vals_end
    cdef np.ndarray[double, ndim=2] res_mat = np.zeros((feat_array.shape[0], dtm.shape[0]))
    cdef long cn, ri, ind, j, k
    cdef double curr_val, ind_doub
    cdef bint at_leaf, found_val
    cdef np.ndarray[np.int32_t, ndim=2] isnan_array = np.isnan(feat_array).astype(np.int32)
    
    # These are in dtm_float
    cdef long THRESH = 0
    cdef long NODE_WEIGHT = 1
    cdef long NODE_VALUE = 2

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


## Below was taken (and modified) from:
# 
# https://shap.readthedocs.io/en/latest/example_notebooks/
# tabular_examples/tree_based_models/Python%20Version%20of%20Tree%20SHAP.html
# 
# extend our decision path with a fraction of one and zero extensions
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef extend_path(np.ndarray[np.int32_t] feature_indexes, 
                np.ndarray[double] zero_fractions,
                np.ndarray[double] one_fractions, 
                np.ndarray[double] pweights,
                long unique_depth, double zero_fraction, 
                double one_fraction, long feature_index):

    cdef long i

    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1
    else:
        pweights[unique_depth] = 0

    for i in range(unique_depth - 1, -1, -1):
        pweights[i+1] += one_fraction * pweights[i] * (i + 1) / (unique_depth + 1)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1)

# undo a previous extension of the decision path
# should try activating cdivision here but wasn't working before
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef unwind_path(np.ndarray[np.int32_t] feature_indexes,
                np.ndarray[double] zero_fractions,
                np.ndarray[double] one_fractions, 
                np.ndarray[double] pweights,
                long unique_depth, long path_index):

    cdef double one_fraction, zero_fraction, next_one_portion, tmp
    cdef long i

    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i+1]
        zero_fractions[i] = zero_fractions[i+1]
        one_fractions[i] = one_fractions[i+1]

# determine what the total permuation weight would be if
# we unwound a previous extension in the decision path
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double unwound_path_sum(np.ndarray[np.int32_t] feature_indexes, 
                     np.ndarray[double] zero_fractions, 
                     np.ndarray[double] one_fractions, 
                     np.ndarray[double] pweights, 
                     long unique_depth,
                     long path_index):
    cdef double one_fraction = one_fractions[path_index]
    cdef double zero_fraction = zero_fractions[path_index]
    cdef double next_one_portion = pweights[unique_depth]
    cdef double total = 0.0
    cdef long i
    cdef double tmp

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            total += tmp;
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1))
        else:
            if zero_fraction == 0:
                print('Warning: zero_fraction is 0')
            else:
                total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1))

    return total

# recursive computation of SHAP values for a decision tree
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef tree_shap_recursive(np.ndarray[np.int32_t] children_left, 
                        np.ndarray[np.int32_t] children_right, 
                        np.ndarray[np.int32_t] children_default, 
                        np.ndarray[np.int32_t] features, 
                        np.ndarray[np.int32_t] node_type_vec, 
                        np.ndarray[np.int32_t] num_cat_vals_vec, 
                        np.ndarray[np.int32_t, ndim=2] cat_vals_mat, 
                        np.ndarray[double] thresholds,
                        np.ndarray[double] values,
                        np.ndarray[double] node_sample_weight,
                        np.ndarray[double] x, 
                        np.ndarray[np.int32_t] x_missing,
                        np.ndarray[double] phi, 
                        long node_index, 
                        long unique_depth, 
                        np.ndarray[np.int32_t] parent_feature_indexes,
                        np.ndarray[double] parent_zero_fractions, 
                        np.ndarray[double] parent_one_fractions, 
                        np.ndarray[double] parent_pweights,
                        double parent_zero_fraction,
                        double parent_one_fraction, 
                        long parent_feature_index, 
                        long condition, 
                        long condition_feature, 
                        double condition_fraction):

    cdef long pfi_len = len(parent_feature_indexes)
    cdef np.ndarray[np.int32_t] feature_indexes = np.zeros(pfi_len, dtype=np.int32)
    cdef np.ndarray[double] zero_fractions = np.zeros(pfi_len, dtype=np.float64)
    cdef np.ndarray[double] one_fractions = np.zeros(pfi_len, dtype=np.float64)
    cdef np.ndarray[double] pweights = np.zeros(pfi_len, dtype=np.float64)

    cdef long i,j,k, split_index, hot_index, cold_index, cleft, cright, path_index,
    cdef double w, hot_zero_fraction, cold_zero_fraction, hot_condition_fraction
    cdef double incoming_zero_fraction, incoming_one_fraction, cold_condition_fraction
    # TODO: optimize below for cython
    # section "extend the unique path" is inefficiently implemented
    # also issues with type consistency (long vs pyint, float v double, etc.)

    # stop if we have no weight coming down to us
    if condition_fraction == 0:
        return

    # # extend the unique path
    # feature_indexes = parent_feature_indexes[unique_depth + 1:]
    # feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    # zero_fractions = parent_zero_fractions[unique_depth + 1:]
    # zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    # one_fractions = parent_one_fractions[unique_depth + 1:]
    # one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    # pweights = parent_pweights[unique_depth + 1:]
    # pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    # extend the unique path
    j=0
    # I am assuming the length of parent_feature_indexes is the same as the others
    for i in range((unique_depth + 1),pfi_len):
        feature_indexes[j] = parent_feature_indexes[i]
        zero_fractions[j] = parent_zero_fractions[i]
        one_fractions[j] = parent_one_fractions[i]
        pweights[j] = parent_pweights[i]
        j+=1
    for i in range(unique_depth+1):   
        feature_indexes[i] = parent_feature_indexes[i]
        zero_fractions[i] = parent_zero_fractions[i]
        one_fractions[i] = parent_one_fractions[i]
        pweights[i] = parent_pweights[i]

    if condition == 0 or condition_feature != parent_feature_index:
        extend_path(
            feature_indexes, zero_fractions, one_fractions, pweights,
            unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth+1):
            w = unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, i)
            phi[feature_indexes[i]] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction
    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif (node_type_vec[node_index]==1) and (x[split_index] < thresholds[node_index]): # (node_type_vec[node_index]==1) and 
            hot_index = cleft
        elif (node_type_vec[node_index]==2):
            hot_index = cright
            for k in range(num_cat_vals_vec[node_index]):
                if x[split_index]==cat_vals_mat[node_index,k]:
                    hot_index = cleft
                    break
        else:
            hot_index = cright
        cold_index = (cright if hot_index == cleft else cleft)
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1
        incoming_one_fraction = 1

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while (path_index <= unique_depth):
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0;
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left, children_right, children_default, features, 
            node_type_vec, num_cat_vals_vec, cat_vals_mat,
            thresholds, values, node_sample_weight,
            x, x_missing, phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        )

        tree_shap_recursive(
            children_left, children_right, children_default, features, 
            node_type_vec, num_cat_vals_vec, cat_vals_mat,
            thresholds, values, node_sample_weight,
            x, x_missing, phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        )
        
def tree_shap_single_pt(dt_mat_int, dt_mat_float,  dt_max_depth, X_in, condition=0, condition_feature=0):

    x_missing = np.isnan(X_in).astype(np.int32)
    # Extract tree info from the matrix/tensor representation
    dt_thresholds = dt_mat_float[:,0]
    dt_node_weights = dt_mat_float[:,1]
    dt_node_vals = dt_mat_float[:,2] # for a single valued output
    
    dt_node_type = dt_mat_int[:,0]
    dt_features = dt_mat_int[:,1]
    dt_left_children = dt_mat_int[:,2]
    dt_right_children = dt_mat_int[:,3]
    dt_na_left = dt_mat_int[:,4]
    dt_default_children = (dt_na_left*dt_left_children + (1-dt_na_left)*dt_right_children).astype(np.int32)
    dt_num_cat_vals = dt_mat_int[:,5]
    dt_cat_vals_mat = dt_mat_int[:,6:]

    # Initialize tracking vectors
    s = (dt_max_depth+2)*(dt_max_depth+1)
    feature_indexes = np.zeros(s, dtype=np.int32)
    zero_fractions = np.zeros(s, dtype=np.float64)
    one_fractions = np.zeros(s, dtype=np.float64)
    pweights = np.zeros(s, dtype=np.float64)
    
    phi = np.zeros(len(X_in)+1) # Assuming single value output
    root_val = (np.sum((dt_node_type==0) * (dt_node_weights) * dt_node_vals) / 
                np.sum((dt_node_type==0) * (dt_node_weights)) )
    # update the bias term, which is the last index in phi
    # (note the paper has this as phi_0 instead of phi_M)
    if condition == 0:
        phi[-1] +=  root_val # Assuming single value output

    # start the recursive algorithm
    tree_shap_recursive(
        dt_left_children, dt_right_children, dt_default_children, dt_features,
        dt_node_type, dt_num_cat_vals, dt_cat_vals_mat,
        dt_thresholds, dt_node_vals, dt_node_weights,
        X_in, x_missing, phi, 0, 0, feature_indexes, zero_fractions, one_fractions, pweights,
        1, 1, -1, condition, condition_feature, 1
    )
    
    return(phi)

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

