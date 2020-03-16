# cython: profile=True

"""Decision Tree Gradient Boosting based on Discrete Graph structure"""
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
from libc.math cimport log as clog
from libc.math cimport exp
from sklearn.metrics import log_loss, mean_squared_error
cimport numpy as cnp
cimport cython


class StructureBoost(object):

    def __init__(self, num_trees, feature_configs,
                 mode='classification', loss_fn='entropy', subsample=1,
                 replace=True, min_size_split=2, min_leaf_size=1,
                 max_depth=3, gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.1):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.dec_tree_list = []
        self.feature_configs = feature_configs
        self._process_feature_configs()
        self.feat_list_full = list(self.feature_configs.keys())
        self.min_size_split = min_size_split
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.feat_sample_by_tree = feat_sample_by_tree
        self.feat_sample_by_node = feat_sample_by_node
        self.loss_fn = loss_fn
        self.subsample = subsample
        self.replace = replace
        self.mode = mode
        if type(loss_fn) == tuple:
            self.loss_fn_der_1 = loss_fn[0]
            self.loss_fn_der_2 = loss_fn[1]
        elif loss_fn == 'entropy':
            self.loss_fn_der_1 = c_entropy_link_der_1
            self.loss_fn_der_2 = c_entropy_link_der_2
        elif loss_fn == 'mse':
            self.loss_fn_der_1 = _mse_der_1
            self.loss_fn_der_2 = _mse_der_2

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

    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
            early_stop_past_steps=0, choose_best_eval=True):

        # Initialize basic info
        if type(y_train)==pd.Series:
            y_train = y_train.values.astype(float)
        elif type(y_train)==np.ndarray:
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
                self.unique_vals_dict[feature] = np.sort(pd.unique(X_train[feature].dropna()))

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
                            min_leaf_size=self.min_leaf_size,
                            gamma=self.gamma, max_depth=self.max_depth,
                            reg_lambda=self.reg_lambda,
                            feat_sample_by_node=self.feat_sample_by_node))
            self.dec_tree_list[i].fit(X_train_to_use, y_g_h_to_use,
                                      feature_sublist=features_for_tree,
                                      uv_dict=self.unique_vals_dict)

    def predict(self, X_test, int num_trees_to_use=0):
        cdef int i
        if num_trees_to_use == 0:
            num_trees_to_use = self.num_trees_for_prediction
        out_vec = self.initial_pred*np.ones(X_test.shape[0])
        for i in range(num_trees_to_use):
            out_vec = (out_vec + self.learning_rate *
                       self.dec_tree_list[i].predict(X_test))
        if self.mode == 'classification':
            return(1/(1+np.exp(-out_vec)))
        else:
            return(out_vec)

    def predict_proba(self, X_test, int num_trees_to_use=0):
        pred_probs = self.predict(X_test, num_trees_to_use)
        return(np.vstack((1-pred_probs,pred_probs)).T)

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
        curr_loss = log_loss(y_true, 1/(1+np.exp(-pred)))
        print("i={}, eval_set_log_loss = {}".format(ind, curr_loss))
    else:
        curr_loss = mean_squared_error(y_true, pred)
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
def c_entropy_link_der_1(cnp.ndarray[double] y_true, cnp.ndarray[double] z_pred):
    cdef int N = y_true.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef double denom1, denom2
    cdef long i
    
    for i in range(N):
        denom_1 = 1+exp(z_pred[i])
        denom_2 = 1+exp(-z_pred[i])
        Y[i]=(-y_true[i]*(1/denom_1) + (1-y_true[i]) * (1/denom_2))
    return Y

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True) 
def c_entropy_link_der_2(cnp.ndarray[double] y_true, cnp.ndarray[double] z_pred):
    cdef int N = y_true.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef double z_pred_exp, minus_z_pred_exp, denom1, denom2
    cdef long i
    
    for i in range(N):
        z_pred_exp = exp(z_pred[i])
        minus_z_pred_exp = exp(-z_pred[i])
        denom_1 = (1+z_pred_exp)*(1+z_pred_exp)
        denom_2 = (1+minus_z_pred_exp)*(1+minus_z_pred_exp)
        Y[i]=(y_true[i]*(z_pred_exp/denom_1) + (1-y_true[i]) * (minus_z_pred_exp/denom_2))
    return Y
