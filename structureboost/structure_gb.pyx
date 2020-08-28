# cython: profile=True

"""Structured Gradient Boosting using graphs"""
import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
from libc.math cimport log as clog
from libc.math cimport exp
cimport numpy as cnp
cimport cython


class StructureBoost(object):
    """Gradient Boosting model allowing categorical structure.

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

    Lucena, B. Proceedings of the Twenty Third International Conference
    on Artificial Intelligence and Statistics, PMLR 108:2949-2958, 2020. 

    Lucena, B. StructureBoost: Efficient Gradient Boosting for Structured
    Categorical Variables. https://arxiv.org/abs/2007.04446
    """
    def __init__(self, num_trees, feature_configs,
                 mode='classification', loss_fn=None, subsample=1,
                 replace=True, min_size_split=2, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.02,
                 random_seed=0, na_unseen_action='weighted_random'):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.dec_tree_list = []
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
        self.replace = replace
        self.random_seed = random_seed
        self.na_unseen_action = na_unseen_action
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
        elif loss_fn == 'entropy':
            self.loss_fn_der_1 = c_entropy_link_der_1
            self.loss_fn_der_2 = c_entropy_link_der_2
        elif loss_fn == 'mse':
            self.loss_fn_der_1 = _mse_der_1
            self.loss_fn_der_2 = _mse_der_2

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
        # Initialize basic info
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if type(y_train) == pd.Series:
            y_train = y_train.to_numpy().astype(float)
        elif type(y_train) == np.ndarray:
            y_train = y_train.astype(float)
        self.eval_freq = eval_freq
        eval_len = np.floor(self.num_trees/self.eval_freq).astype(int)
        self.eval_results = np.zeros(eval_len)
        if self.mode == 'classification':
            prob_est = np.mean(y_train)
            self.initial_pred = np.log(prob_est / (1-prob_est))
        else:
            self.initial_pred = np.mean(y_train)
        num_rows = X_train.shape[0]
        col_list = list(X_train.columns)
        column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        curr_answer = self.initial_pred * np.ones(len(y_train))
        stop_now = False

        # Initalize unique values dict
        self.unique_vals_dict = {}
        for feature in self.feature_configs.keys():
            if self.feature_configs[feature]['feature_type'] == 'numerical':
                self.unique_vals_dict[feature] = np.sort(
                                    pd.unique(X_train[feature].dropna()))

        # Initalize eval_set related
        if eval_set is not None:
            X_valid = eval_set[0]
            y_valid = eval_set[1].astype(float)
            curr_valid_answer = self.initial_pred * np.ones(len(y_valid))
            curr_valid_loss = _output_loss(y_valid, curr_valid_answer,
                                           0, self.mode)

        # Main loop to build trees
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
                    if ((i+1) % self.eval_freq == 1):
                        curr_loss = _output_loss(y_valid, curr_valid_answer,
                                                 i, self.mode)
                        curr_step = np.floor((i+1) /
                                             self.eval_freq).astype(int)-1
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
            y_g_vec = self.loss_fn_der_1(y_train, curr_answer)
            y_h_vec = self.loss_fn_der_2(y_train, curr_answer)
            y_g_h_mat = np.vstack((y_g_vec, y_h_vec)).T

            # Sample the data to use for this tree
            rows_to_use = _get_rows_for_tree(num_rows, self.subsample,
                                             self.replace)

            # Determine which features to consider for this node
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
            self.dec_tree_list.append(stdt.StructureDecisionTree(
                            feature_configs=self.feature_configs,
                            feature_graphs=self.feature_graphs,
                            min_size_split=self.min_size_split,
                            gamma=self.gamma, max_depth=self.max_depth,
                            reg_lambda=self.reg_lambda,
                            feat_sample_by_node=self.feat_sample_by_node))
            self.dec_tree_list[i].fit(X_train_to_use, y_g_h_to_use,
                                      feature_sublist=features_for_tree,
                                      uv_dict=self.unique_vals_dict)
        if self.na_unseen_action == 'weighted_random':
            self.rerandomize_na_dir_weighted_all_trees()

    def predict(self, X_test, int num_trees_to_use=-1):
        """Returns a one-dimensional response - probability of 1 or numeric prediction

        If mode is classification (which is binary classification) then
        predict returns the probability of being in class '1'. (Note: unlike
        sklearn and other packages, `predict` will **not** return a 
        hard 1/0 prediction.  For `hard` prediction of class membership,
        threshold the probabilities appropriately.)

        If mode is regression, then predict returns the point estimate of
        the numerical target.

        Use `predict_proba` to get a 2-d matrix of class probabilities.

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

        out_vec : array-like
            Returns a numpy array of length n, where n is the number of rows
            in X-test.
        """
        cdef int i
        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        out_vec = self.initial_pred*np.ones(X_test.shape[0])
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


def _get_rows_for_tree(num_rows, subsample, replace):
    if (subsample == 1) and (not replace):
        return np.arange(num_rows)
    rows_to_return = int(np.ceil(num_rows*subsample))
    if replace:
        return np.random.randint(0, num_rows, rows_to_return)
    else:
        return np.random.choice(num_rows, rows_to_return, replace=False)


def _output_loss(y_true, pred, ind, mode):
    if mode == 'classification':
        curr_loss = my_log_loss(y_true, 1/(1+np.exp(-pred)))
        print("i={}, eval_set_log_loss = {}".format(ind, curr_loss))
    else:
        curr_loss = my_mean_squared_error(y_true, pred)
        print("i={}, eval_set_mse = {}".format(ind, curr_loss))
    return curr_loss


def _entropy_der_1(y_true, y_pred, eps=1e-15):
    y_pred = np.maximum(y_pred, eps)
    y_pred = np.minimum(y_pred, 1-eps)
    return((-(y_true/y_pred) + (1-y_true)/(1-y_pred)))


def _entropy_der_2(y_true, y_pred, eps=1e-15):
    y_pred = np.maximum(y_pred, eps)
    y_pred = np.minimum(y_pred, 1-eps)
    out_vec = (y_true)/(y_pred**2) + ((1-y_true)/((1-y_pred)**2))
    return(out_vec)


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
    # return(y_true*(np.exp(z_pred)/((1+np.exp(z_pred))**2)) +
    #        (1-y_true) * (np.exp(-z_pred)/((1+np.exp(-z_pred))**2)))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def c_entropy_link_der_1(cnp.ndarray[double] y_true,
                         cnp.ndarray[double] z_pred):
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
def c_entropy_link_der_2(cnp.ndarray[double] y_true,
                         cnp.ndarray[double] z_pred):
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


def my_log_loss(y_true, y_pred, eps=10e-16):
    y_pred = np.minimum(y_pred, (1-eps))
    y_pred = np.maximum(y_pred, eps)
    out_val = -np.mean(y_true*(np.log(y_pred)) + (1-y_true)*np.log(1-y_pred))
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
