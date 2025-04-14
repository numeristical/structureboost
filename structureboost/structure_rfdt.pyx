# cython: profile=True
# cython: language_level=3

"""Decision Tree based on Discrete Graph structure"""
import graphs
import copy
import random
import warnings
import numpy as np
import scipy as sp
import pandas as pd
from structure_dt import StructureDecisionTree
from libc.math cimport log as clog
from libc.math cimport isnan
cimport numpy as np
np.import_array()
cimport cython

# ctypedef np.int64_t dtype_int64_t

class StructureRFDecisionTree(StructureDecisionTree):
    """Decision Tree using graphical structure: variant for Random Forest
    """

    def __init__(self, feature_configs, feature_graphs, min_size_split=2,
                 max_depth=50, feat_sample_by_node=1, mode='classification',
                 min_leaf_size=None):
        super().__init__(feature_configs=feature_configs,
                         feature_graphs=feature_graphs,
                         min_size_split=min_size_split,
                         max_depth=max_depth,
                         gamma=0,
                         feat_sample_by_node=feat_sample_by_node,
                         reg_lambda=1)
        self.mode=mode
        self.min_leaf_size = min_leaf_size
        if min_leaf_size is None:
            if self.mode=='regression':
                self.min_leaf_size=5
            elif self.mode=='classification':
                self.min_leaf_size=1
        else:
            self.min_leaf_size = min_leaf_size


    def _check_stopping_condition(self, num_dp, depth, g_h_train_node):
        num_unique_vals = len(pd.unique(g_h_train_node[:,0]))
        cond = ((num_dp < self.min_size_split) or
                       (depth >= self.max_depth)
                        or (num_unique_vals<=1))
        return cond


    def _node_summary_gh(self, y_c_mat):
        if (y_c_mat.shape[0] == 0):
            # This shouldn't happen
            return 0
        else:
            out_val = np.mean(y_c_mat[:,0])
            return(out_val)

    def _evaluate_numerical_splits(self, feature_vec, y_c_mat,
                                   split_vec):

        has_na_vals = np.isnan(split_vec[-1])
        bin_result_vec = np.searchsorted(split_vec,
                                         feature_vec,
                                         side='right').astype(np.int32)
        y_sum_bins, count_in_bins = get_bin_sums_c_rfbin(y_c_mat[:,0],
                                                bin_result_vec,
                                                len(split_vec)+1)
        yst, ysl, ysr, cbt, cbl, cbr =  get_left_right_sums_rf(
                                            y_sum_bins, count_in_bins)
        if self.mode=='classification':
            score_vec = _get_score_array_binary_entropy(ysl, ysr, cbl, cbr)
            mask = (cbl<self.min_leaf_size) | (cbr<self.min_leaf_size)
            score_vec[mask] = np.inf

        elif self.mode=='regression':
            score_vec = _get_score_array_mse(y_c_mat[:,0], bin_result_vec,
                                                    ysl, ysr, cbl, cbr)
            mask = (cbl<self.min_leaf_size) | (cbr<self.min_leaf_size)
            score_vec[mask] = np.inf

        best_loss, best_split_val = _get_best_vals(score_vec, split_vec)
        if has_na_vals and (len(split_vec) > 2):
            ysl_nal = (ysl + (yst - ysl[-1]))[:-1]
            ysr_nal = (ysr - (yst - ysl[-1]))[:-1]
            cbl_nal = (cbl + (cbt - cbl[-1]))[:-1]
            cbr_nal = (cbr - (cbt - cbl[-1]))[:-1]

            if self.mode=='classification':
                score_vec_nal = _get_score_array_binary_entropy(ysl_nal, ysr_nal,
                                                               cbl_nal, cbr_nal)
                mask = (cbl_nal<self.min_leaf_size) | (cbr_nal<self.min_leaf_size)
                score_vec_nal[mask] = np.inf
            elif self.mode=='regression':
                score_vec_nal = _get_score_array_mse(y_c_mat[:,0], bin_result_vec,
                                                        ysl_nal, ysr_nal,
                                                        cbl_nal, cbr_nal)
                mask = (cbl_nal<self.min_leaf_size) | (cbr_nal<self.min_leaf_size)
                score_vec_nal[mask] = np.inf

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

    def _augment_to_y_c_for_rf(self, y_train):
        """For"""
        return np.vstack((y_train, np.ones(y_train.shape[0]))).T


    def get_loss_in_span_tree(self, feature_vec_node, y_c_mat, y_c_accum_array,
                              y_c_sum, leaf_vertex_ind, left_feat_values,
                                                   feature_type):
        y_left = y_c_accum_array[leaf_vertex_ind,0]
        c_left = y_c_accum_array[leaf_vertex_ind,1]
        if self.mode=='classification':
            curr_loss = _get_score_binary_entropy(y_left, y_c_sum[0]-y_left,
                                                        c_left, y_c_sum[1]-c_left,
                                                        float(self.min_leaf_size))

        elif self.mode=='regression':
            if feature_type=='categorical_str':
                mask = get_mask(feature_vec_node,left_feat_values)
                curr_loss = _get_mse_mask(y_c_mat[:,0], mask, y_left,
                                            y_c_sum[0]-y_left,
                                            c_left, y_c_sum[1]-c_left,
                                            float(self.min_leaf_size))
            elif feature_type in ['categorical_int','graphical_voronoi']:
                fs_array = np.fromiter(left_feat_values, np.int32,
                len(left_feat_values)).astype(np.int32)
                vec_len = len(feature_vec_node)
                lsplit_len = len(fs_array)
                mask = np.zeros(vec_len, dtype=np.int32)
                mask = get_mask_int_c(feature_vec_node.astype(np.int32),
                               fs_array, vec_len, lsplit_len,
                               mask)

                curr_loss = _get_mse_mask(y_c_mat[:,0], mask, y_left,
                                            y_c_sum[0]-y_left,
                                            c_left, y_c_sum[1]-c_left,
                                            float(self.min_leaf_size))
        return(curr_loss)


    def get_score_of_split(self, y_c_mat, mask_left, y_c_sum):
        y_c_mat_left = y_c_mat[mask_left,:]
        y_c_left = np.sum(y_c_mat_left, axis=0)
        y_c_right = y_c_sum-y_c_left
        if self.mode=='classification':
            loss = _get_score_binary_entropy(y_c_left[0], y_c_right[0],
                                                    y_c_left[1], y_c_right[1],
                                                    float(self.min_leaf_size))
            return(loss)
        elif self.mode=='regression':
            loss = _get_mse_mask(y_c_mat[:,0], mask_left, y_c_left[0], y_c_right[0],
                                y_c_left[1], y_c_right[1],
                                float(self.min_leaf_size))
        return(loss)




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_bin_sums_c_rfbin(np.ndarray[double] y_vec,
                   np.ndarray[np.int32_t] bin_result_vec, long out_vec_size):
    cdef int i
    cdef int m = bin_result_vec.shape[0]

    cdef np.ndarray[double] y_sum_bins = np.zeros(out_vec_size)
    cdef np.ndarray[double] count_in_bins = np.zeros(out_vec_size)

    for i in range(m):
        y_sum_bins[bin_result_vec[i]] += y_vec[i]
        count_in_bins[bin_result_vec[i]] += 1
    return y_sum_bins, count_in_bins

def get_left_right_sums_rf(y_bin_sums, count_in_bins):
    y_sum_left = np.cumsum(y_bin_sums)
    y_sum_total = y_sum_left[-1]
    y_sum_left = y_sum_left[:-1]
    y_sum_right = y_sum_total - y_sum_left
    count_in_bins_left = np.cumsum(count_in_bins)
    count_in_bins_total = count_in_bins_left[-1]
    count_in_bins_left = count_in_bins_left[:-1]
    count_in_bins_right = count_in_bins_total - count_in_bins_left
    return (y_sum_total, y_sum_left, y_sum_right, count_in_bins_total,
            count_in_bins_left, count_in_bins_right)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def _get_score_array_binary_entropy(np.ndarray[double] ysl, 
                                    np.ndarray[double] ysr, 
                                    np.ndarray[double] cbl, 
                                    np.ndarray[double] cbr, 
                                    double eps=1e-15):
    cdef np.ndarray[double] prob_of_left = cbl/(cbl+cbr)
    cdef np.ndarray[double] prob_of_right = 1-prob_of_left
    cdef np.ndarray[double] prob_left = ((ysl)/(cbl+eps))
    cdef np.ndarray[double] prob_right = ((ysr)/(cbr+eps))
    cdef np.ndarray[double] prob_left_clip = np.clip(prob_left, eps, 1-eps)
    cdef np.ndarray[double] prob_right_clip = np.clip(prob_right, eps, 1-eps)
    cdef np.ndarray[double] entropy_left = -prob_left*np.log(prob_left_clip) - (1-prob_left)*np.log(1-prob_left_clip)
    cdef np.ndarray[double] entropy_right = -prob_right*np.log(prob_right_clip) - (1-prob_right)*np.log(1-prob_right_clip)
    cdef np.ndarray[double] score_vec = prob_of_left*entropy_left + prob_of_right*entropy_right
    return(score_vec)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def _get_score_binary_entropy(double ysl, 
                                    double ysr, 
                                    double cbl, 
                                    double cbr, 
                                    double mls,
                                    double eps=1e-15):
    cdef double prob_of_left, prob_of_right, prob_left, prob_right
    cdef double prob_left_clip, prob_right_clip, entropy_left, entropy_right

    if ((cbl<mls) or (cbr<mls)):
        return np.inf
    else:
        prob_of_left = cbl/(cbl+cbr)
        prob_of_right = 1-prob_of_left
        prob_left = ((ysl)/(cbl+eps))
        prob_right = ((ysr)/(cbr+eps))
        prob_left_clip = max(eps, min(prob_left, 1-eps))
        prob_right_clip = max(eps, min(prob_right, 1-eps))
        entropy_left = -prob_left*np.log(prob_left_clip) - (1-prob_left)*np.log(1-prob_left_clip)
        entropy_right = -prob_right*np.log(prob_right_clip) - (1-prob_right)*np.log(1-prob_right_clip)
        return(prob_of_left*entropy_left + prob_of_right*entropy_right)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def _get_score_array_mse(np.ndarray[double] y_vec, 
                        np.ndarray[np.int32_t] bin_result_vec, 
                        np.ndarray[double] ysl, 
                        np.ndarray[double] ysr, 
                        np.ndarray[double] cbl, 
                        np.ndarray[double] cbr, 
                        double eps=1e-15):

    cdef np.ndarray[double] score_vec = np.zeros(len(ysl))
    cdef np.ndarray[double] adj_y_vec = np.zeros(len(y_vec))
    cdef long i,j
    cdef double left_mean, right_mean
    for i in range(len(score_vec)):
        left_mean = ysl[i]/(cbl[i]+eps)
        right_mean = ysr[i]/(cbr[i]+eps)
        for j in range(len(y_vec)):
            if bin_result_vec[j]<=i:
                adj_y_vec[j] = y_vec[j] - left_mean
            else:
                adj_y_vec[j] = y_vec[j] - right_mean
        score_vec[i] = np.sum(adj_y_vec * adj_y_vec)
    return score_vec

def _get_best_vals(score_vec, split_vec):
    best_split_index = np.argmin(score_vec)
    best_loss = score_vec[best_split_index]
    best_split_val = split_vec[best_split_index]
    return best_loss, best_split_val

# def _get_mse_st(y_vec, mask, ysl, ysr, cbl, cbr, eps=1e-15):
#     adj_y_vec = np.zeros(len(y_vec))
#     left_mean = ysl/(cbl+eps)
#     right_mean = ysr/(cbr+eps)
#     anti_mask = np.logical_not(mask)
#     adj_y_vec[mask] = y_vec[mask] - left_mean
#     adj_y_vec[anti_mask] = y_vec[anti_mask] - right_mean
#     loss = np.sum(adj_y_vec * adj_y_vec)
#     return(loss)

# def _get_mse_st_int(y_vec, feature_vec_node, left_feat_values, ysl, ysr, cbl, cbr, eps=1e-15):
#     adj_y_vec = np.zeros(len(y_vec))
#     left_mean = ysl/(cbl+eps)
#     right_mean = ysr/(cbr+eps)
#     # fs_array = np.fromiter(left_feat_values, int,
#     #                        len(left_feat_values)).astype(np.int64)
#     # vec_len = len(feature_vec_node)
#     # lsplit_len = len(fs_array)
#     # mask = np.zeros(vec_len, dtype=np.int64)
#     # mask = get_mask_int_c(feature_vec_node.astype(np.int64),
#     #                                fs_array, vec_len, lsplit_len,
#     #                                mask)
#     anti_mask = np.logical_not(mask)
#     adj_y_vec[mask] = y_vec[mask] - left_mean
#     adj_y_vec[anti_mask] = y_vec[anti_mask] - right_mean
#     loss = np.sum(adj_y_vec * adj_y_vec)
#     return(loss)

def get_mask(feature_vec_node, left_split):
    return np.array([x in left_split for x in feature_vec_node])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_mask_int_c(np.ndarray[np.int32_t] feature_vec_node,
                   np.ndarray[np.int32_t] left_split,
                   long vec_len, long lsplit_len,
                   np.ndarray[np.int32_t] mask_vec):
    cdef int i, j
    for i in range(vec_len):
        for j in range(lsplit_len):
            if feature_vec_node[i] == left_split[j]:
                mask_vec[i] = 1
                break
    return mask_vec.astype(bool)


def _get_mse_mask(y_vec, mask, ysl, ysr, cbl, cbr, mls, eps=1e-15):
    if ((cbl<mls) or (cbr<mls)):
        return np.inf
    else:
        adj_y_vec = np.zeros(len(y_vec))
        left_mean = ysl/(cbl+eps)
        right_mean = ysr/(cbr+eps)
        anti_mask = np.logical_not(mask)
        adj_y_vec[mask] = y_vec[mask] - left_mean
        adj_y_vec[anti_mask] = y_vec[anti_mask] - right_mean
        loss = np.sum(adj_y_vec * adj_y_vec)
    return(loss)

