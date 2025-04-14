# cython: profile=True
# cython: language_level=3

"""Multi-class Structured Gradient Boosting using graphs"""
import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
from structure_gb import StructureBoost
from libc.math cimport log as clog
from libc.math cimport exp
from structure_dt_multi import StructureDecisionTreeMulti
from scipy.special import softmax
cimport numpy as np
np.import_array()
cimport cython


class StructureBoostMulti(StructureBoost):
    """Gradient Boosting model allowing categorical structure in both
    predictor and target variables.

    For examples on configuration, see:
        www.github.com/numeristical/structureboost/examples

    Categorical predictor variables require a graph as part of their feature config.

    To incorporate structure in a target variable, you must supply either
    a partition (and associated weights), or a graph (and other parameters)

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

    num_classes : int
        The number of classes in the multiclass classification problem.

    target_structure : dict, Default is None
        A dictionary that contains the structure to be used.  This can be a
        fixed set of partitions with corresponding weights or a graph
        together with parameters determining how to (randomly) choose the
        partitions at each step (and how many to choose). See documentation
        at https://structureboost.readthedocs.io/ for details on how to
        configure this.  Default is None which will train a "standard"
        multiclass model without any structural information.

    learning_rate: float, default is .02
        The "step size" to take when adding each new tree to the model.
        We recommend using a relatively low learning rate combined with
        early stopping.

    max_depth : int, default is 3
        The maximum depth to use when building trees. This is an important
        parameter.  Smaller values are more likely to underfit and 
        larger values more likely to overfit.

    loss_fn : 'entropy' or tuple of functions
        Loss fn to use.  Default is 'entropy' (aka log-loss aka cross-entropy).
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
    def __init__(self, num_trees,
                 num_classes,  feature_configs=None,
                 target_structure=None,
                 loss_fn=None, subsample=1,
                 initial_model=None,
                 replace=True, min_size_split=25, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.02,
                 random_seed=0, na_unseen_action='weighted_random',
                 prec_digits=6, default_configs=None):
        super().__init__(num_trees, feature_configs,
                 'classification',
                 loss_fn, subsample,
                 initial_model,
                 replace, min_size_split, max_depth,
                 gamma, reg_lambda, feat_sample_by_tree,
                 feat_sample_by_node, learning_rate,
                 random_seed, na_unseen_action, prec_digits, default_configs)
        self.num_classes = num_classes
        self.ts_dict = target_structure
        if self.ts_dict is not None:
            self._process_target_structure()

    def _process_target_structure(self):
        pt = self.ts_dict['partition_type']
        if (pt == 'fixed'):
            self._process_fixed_partition()
        elif (pt in ['variable','random']):
            self._process_variable_partition()
            # Generate first random partition
        else:
            warnings.warn('Unknown partition_type in target_structure')

    def _process_variable_partition(self):
        self._process_singleton_weight()
        self._process_target_graph()
        if 'num_partitions' in self.ts_dict.keys():
            self.num_partitions = self.ts_dict['num_partitions']
        else:
            self.num_partitions = 1
            print("'num_partitions' not configured. Defaulting to 1.")
        if 'random_partition_size' in self.ts_dict.keys():
            self.random_partition_size = self.ts_dict['random_partition_size']
        if (self.num_partitions==1) and (type(self.random_partition_size)!= int):
            w_str = "'random_partition_size' should be an integer when "
            w_str+= "'num_partitions' equals 1"
            warnings.warn(w_str)
        if (self.num_partitions>1) and (len(self.random_partition_size)!=
                                        self.num_partitions):
            w_str = "'random_partition_size' should have length equal to "
            w_str+= "'num_partitions'."
            warnings.warn(w_str)
        self.part_weight_vec =(((1-self.singleton_weight)/self.num_partitions) * 
                                np.ones(self.num_partitions))
        if 'rp_method' in self.ts_dict.keys():
            self.rp_method = self.ts_dict['rp_method']
            if self.rp_method not in ['span_tree', 'contraction']:
                w_str = "Unknown 'rp_method'. Defaulting to 'span_tree'"
                warnings.warn(w_str)
                self.rp_method='span_tree'
        else:
            print("'rp_method' not configured. Defaulting to 'span_tree'")
            self.rp_method='span_tree'

    def _generate_random_partitions(self):
        if self.num_partitions==1:
            rp_list = [self.ts_dict['target_graph'].random_partition(
                        self.random_partition_size, self.rp_method)]
        else:
            rp_list = [self.ts_dict['target_graph'].random_partition(
                        self.random_partition_size[i], self.rp_method) 
                        for i in range(len(self.random_partition_size))]
        self.rpt = self._create_rpt_from_list(rp_list)

    def _process_fixed_partition(self):
        self._process_singleton_weight()
        self.singleton_weight = self.ts_dict['singleton_weight']
        self.part_weight_vec = np.array(self.ts_dict['partition_weight_vec'])
        all_g_0 = np.all(self.part_weight_vec>0)
        if not all_g_0:
            w_str="Some weights are not positive - may cause unexpected results"
            warnings.warn(w_str)
        w_sum = np.sum(self.part_weight_vec)+self.ts_dict['singleton_weight']
        if not np.allclose(w_sum,1):
            w_str="Sum of partition weights and singleton weight is {}".format(w_sum)
            w_str+=" instead of 1. May cause unexpected results."
            warnings.warn(w_str)
        self.rpt = self._create_rpt_from_list(self.ts_dict['partition_list'])
        part_sums = np.sum(self.rpt, axis=1)
        exhaust_mut_exc = np.all(part_sums == np.ones(part_sums.shape))
        if not exhaust_mut_exc:
            w_str="One or more partitions are either not exhaustive or "
            w_str+="or not mutually exclusive. May cause unexpected results."
            warnings.warn(w_str)

    def _process_singleton_weight(self):
        if 'singleton_weight' in self.ts_dict.keys():
            self.singleton_weight = self.ts_dict['singleton_weight']
        else:
            w_str = "'singleton_weight' not configured in target_structure."
            w_str = "Using weight of 0.5"
            self.singleton_weight = .5
            warnings.warn(w_str)
        if (self.singleton_weight<0) or (self.singleton_weight>1):
            w_str = "'singleton_weight' should be between 0 and 1."
            w_str+= " May cause unexpected results."
            warnings.warn(w_str)

    def _process_target_graph(self):
        if 'target_graph' not in self.ts_dict.keys():
            w_str = "target_graph not configured"
            warnings.warn(w_str)
        else:
            self.target_graph = self.ts_dict['target_graph']
            if (frozenset(self.target_graph.vertices)!=
                                frozenset(np.arange(self.num_classes))):
                w_str = "Vertices of graph should be the integers 0,1,...,num_classes-1"
                w_str+= ". May cause unexpected results if this is not the case."
                warnings.warn(w_str)


    def _create_rpt_from_list(self, partition_list):
        num_part = len(partition_list)
        max_part_size = np.max(np.array([len(qq) for qq in partition_list]))
        rpt = np.zeros((num_part, max_part_size, self.num_classes), dtype=np.int32)
        flat_list = [j for sl in partition_list for i in sl for j in i]
        min_val, max_val = np.min(flat_list), np.max(flat_list)
        if (min_val<0) or (max_val>self.num_classes-1):
            w_str = "Elements of partition must be >=0 and <=(num_classes-1). "
            w_str+= "Results may be unexpected."
            warnings.warn(w_str)
        for i in range(num_part):
            for j in range(len(partition_list[i])):
                for k in partition_list[i][j]:
                        rpt[i,j,k]=1
        return(rpt)

    def _process_y_data(self, y_data):
        if type(y_data) == pd.Series:
            y_data = y_data.to_numpy().astype(np.int32)
        elif type(y_data) == np.ndarray:
            y_data = y_data.astype(np.int32)
        #y_data = get_one_hot_mat(y_data, self.num_classes)
        return(y_data)

    def _get_initial_pred(self, X_train, y_train):
        if self.initial_model is not None:
            start_probs = self.initial_model.predict_proba(X_train)
            start_probs = np.clip(start_probs, 1e-15, 1-1e-15)
            return(np.log(start_probs))
        else:
            # Improve error checking on values of y_train
            # Also, should they be forced to ints?
            prob_est_vec = np.clip(np.mean(get_one_hot_mat(y_train, self.num_classes), axis=0), 1e-15, 1-1e-15)
            return(np.tile(np.log(prob_est_vec), (X_train.shape[0],1)))

    def _add_train_next_tree(self, features_for_tree, X_train, y_g_h_train, index):
        self.dec_tree_list.append(StructureDecisionTreeMulti(
                        feature_configs=self.feature_configs,
                        feature_graphs=self.feature_graphs,
                        num_classes=self.num_classes,
                        min_size_split=self.min_size_split,
                        gamma=self.gamma, max_depth=self.max_depth,
                        reg_lambda=self.reg_lambda,
                        feat_sample_by_node=self.feat_sample_by_node))
        self.dec_tree_list[index].fit(X_train, y_g_h_train,
                                  feature_sublist=features_for_tree,
                                  uv_dict=self.unique_vals_dict)


    def process_curr_answer(self, curr_answer):
        max_phi_vec = np.max(curr_answer, axis=1).reshape(-1,1)
        curr_answer = curr_answer - max_phi_vec
        return curr_answer


    def compute_gh_mat(self, y_train, curr_answer, index):
        if self.ts_dict is None:
            exp_phi_mat=np.exp(curr_answer)
            exp_phi_mat_sum_vec=np.sum(exp_phi_mat, axis=1)
            y_g_h_mat = c_entropy_link_der_12_vec_sp(y_train, exp_phi_mat, exp_phi_mat_sum_vec)
            return y_g_h_mat
        else:
            if (self.ts_dict['partition_type'] == 'variable'):
                self._generate_random_partitions()
            exp_phi_mat=np.exp(curr_answer)
            exp_phi_mat_sum_vec=np.sum(exp_phi_mat, axis=1)
            y_g_h_mat = c_str_entropy_link_der_12_vec_sp(y_train, exp_phi_mat,
                                        exp_phi_mat_sum_vec,
                                        self.part_weight_vec, self.rpt)
            if self.singleton_weight>0:
                y_g_h_mat_reg = c_entropy_link_der_12_vec_sp(y_train, exp_phi_mat, exp_phi_mat_sum_vec)
                y_g_h_mat = y_g_h_mat + self.singleton_weight * y_g_h_mat_reg
            return y_g_h_mat


    def _compute_loss(self, y_true, pred):
        return(my_log_loss_vec(get_one_hot_mat(y_true, self.num_classes),
                               self.softmax_mat(pred)))


    def _predict_py(self, X_test, int num_trees_to_use=-1):
        cdef int i
        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        if self.initial_model is None:
            # todo: revisit below
            out_mat = np.tile(self.initial_pred[0,:],(X_test.shape[0],1))
        else:
            out_mat = self.initial_model.predict_proba(X_test)
            out_mat = np.clip(out_mat, 1e-15, 1-1e-15)
            out_mat = np.log(out_mat)
        for i in range(num_trees_to_use):
            out_mat = (out_mat + self.learning_rate *
                       self.dec_tree_list[i].predict(X_test))
        return(self.softmax_mat(out_mat))

    def _predict_fast(self, X_test, int num_trees_to_use=-1):

        if not self.optimized:
            if self.optimizable:
                self.get_tensors_for_predict()
            else:
                print('Fast predict not possible for this model type')
                return None

        cdef int i
        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        if self.initial_model is None:
            # todo: revisit below
            out_mat = np.tile(self.initial_pred[0,:],(X_test.shape[0],1))
        else:
            out_mat = self.initial_model.predict_proba(X_test)
            out_mat = np.clip(out_mat, 1e-15, 1-1e-15)
            out_mat = np.log(out_mat)
        tree_pred_tens = predict_with_tensor_c_mc(
                                self.pred_tens_float[:num_trees_to_use,:,:],
                                self.pred_tens_int[:num_trees_to_use,:,:],
                                X_test, self.num_classes)
        out_mat = out_mat + self.learning_rate*np.sum(tree_pred_tens, axis=1)
        return(self.softmax_mat(out_mat))

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
        return(self.predict(X_test, num_trees_to_use))


    # def softmax_mat(self, phi_mat):
    #     max_phi_mat = np.max(phi_mat) #.reshape(-1,1)
    #     phi_mat = phi_mat - max_phi_mat
    #     exp_mat = np.exp(phi_mat)
    #     exp_sum_vec = np.sum(exp_mat, axis=1)
    #     return(exp_mat/(exp_sum_vec.reshape(-1,1)))

    def softmax_mat(self, phi_mat):
        return(softmax(phi_mat, axis=1))

    def get_tensors_for_predict(self):
        if self.optimizable:
            cat_size = np.max(np.array([dt.get_max_split_size() for dt in self.dec_tree_list]))
            num_dt = len(self.dec_tree_list)
            max_nodes = np.max(np.array([dt.num_nodes for dt in self.dec_tree_list]))
            self.pred_tens_int = np.zeros((num_dt, max_nodes, cat_size+6), dtype=np.int32)-1
            self.pred_tens_float = np.zeros((num_dt, max_nodes, self.num_classes+2))
            for i in range(num_dt):
                self.convert_dt_to_matrix(i)
            self.optimized=True
        else:
            print("Model not optimizable for predict due to string or voronoi variable.")

    # # These are in dtm_float
    # cdef long THRESH = 0
    # cdef long NODE_WEIGHT = 1
    # cdef long NODE_VALUE_START = 2
    # vector output from 2 to num_classes+1 inclusive

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
            self.pred_tens_float[dt_num, ni, 1] = float(node['num_data_points'])
            self.pred_tens_float[dt_num, ni, 2:self.num_classes+2] = node['node_summary_val']
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

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_with_tensor_c_mc(np.ndarray[double, ndim=3] dtm_float,
                      np.ndarray[np.int32_t, ndim=3] dtm,
                      np.ndarray[double, ndim=2] feat_array,
                      long num_classes):
    
    cdef long cat_vals_end
    cdef np.ndarray[double, ndim=3] res_tens = np.zeros((
                                feat_array.shape[0], dtm.shape[0], num_classes))
    cdef long cn, ri, ind, j, k, q
    cdef double curr_val, ind_doub
    cdef bint at_leaf, found_val
    cdef np.ndarray[np.int32_t, ndim=2] isnan_array = np.isnan(feat_array).astype(np.int32)
    
    # These are in dtm_float
    cdef long THRESH = 0
    cdef long NODE_WEIGHT = 1
    cdef long NODE_VALUE_START = 2

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
                    for q in range(num_classes):
                        res_tens[ri,k,q] = dtm_float[k,cn,NODE_VALUE_START+q]
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
    return(res_tens)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def c_entropy_link_der_12_vec_sp(np.ndarray[np.int32_t] y_true,
                         double[:,:] exp_phi_pred,
                         double[:] exp_phi_sum_vec):
    """In this variant, y_true is just a vector of correct classes (not one hot)"""
    cdef long N = len(y_true)
    cdef long m = exp_phi_pred.shape[1]
    cdef double[:,:] Y = np.zeros((N,m+m))
    cdef long i,j,k

    for i in range(N):
        for j in range(m):
            if y_true[i]==j:
                Y[i,j]=(exp_phi_pred[i,j]/exp_phi_sum_vec[i])-1
            else:
                Y[i,j]=(exp_phi_pred[i,j]/exp_phi_sum_vec[i])
            Y[i,j+m]=-(exp_phi_pred[i,j]*(exp_phi_pred[i,j]-exp_phi_sum_vec[i])/
                                             (exp_phi_sum_vec[i]*exp_phi_sum_vec[i]))
    return np.asarray(Y)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def c_str_entropy_link_der_12_vec_sp(np.ndarray[np.int32_t] y_true,
                                 double[:,:] exp_phi_pred,
                                 double[:] exp_phi_sum_vec, 
                                 double[:] weight_vec, 
                                 np.int32_t[:,:,:] rp_tensor):
    cdef long N = len(y_true)
    cdef long m = exp_phi_pred.shape[1]
    cdef long ind = 0
    cdef long qqq, xyz
    cdef long yval = 0
    cdef double curr_wt
    cdef double[:,:] Y = np.zeros((N,2*m))
    cdef double all_sum = 0
    cdef double set_sum = 0
    cdef long num_part = rp_tensor.shape[0]
    cdef long i,j,k,t
    cdef double jt, jk

    for qqq in range(num_part):
        curr_wt = weight_vec[ind]
        for i in range(N):
            xyz=0
            # get y_true value from y_true onehot (passed in)
            yval = y_true[i]
            while (rp_tensor[qqq,xyz,yval]==0):
                xyz+=1
            set_sum = 0
            for t in range(m):
                if (rp_tensor[qqq,xyz,t]==1):
                    set_sum+=exp_phi_pred[i,t]
            for j in range(m):
                jt = (exp_phi_pred[i,j]/set_sum)
                jk = (exp_phi_pred[i,j]/exp_phi_sum_vec[i])
                # if j in set containing y_true
                if (rp_tensor[qqq,xyz,j]==1):
                    #Y[i,j]+= value when j in T * weight
                    Y[i,j]+=(-curr_wt * (exp_phi_pred[i,j]*(exp_phi_sum_vec[i]-set_sum)/
                                        (set_sum*exp_phi_sum_vec[i])))
                    Y[i,j+m]+=(curr_wt * (jt-jk+jk*jk-jt*jt))
                else:
                    #Y[i,j]+= value when j not in T * weight
                    Y[i,j]+=curr_wt * exp_phi_pred[i,j]/exp_phi_sum_vec[i]
                    Y[i,j+m]+=(-curr_wt * (-jk+jk*jk))
    return np.asarray(Y)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def get_one_hot_mat(np.ndarray[np.int32_t] y_true,
                         int num_classes):
    cdef int N = y_true.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] Y = np.zeros((N,num_classes), dtype=np.int32)
    
    for i in range(N):
        Y[i,y_true[i]]= 1
    return(Y)


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

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_entropy_link_der_12_vec(np.ndarray[dtype_int64_t, ndim=2] y_true,
#                          double[:,:] exp_phi_pred,
#                          double[:] exp_phi_sum_vec):
#     cdef long N = y_true.shape[0]
#     cdef long m = exp_phi_pred.shape[1]
#     cdef double[:,:] Y = np.zeros((N,m+m))
#     cdef long i,j,k

#     for i in range(N):
#         for j in range(m):
#             if y_true[i,j]==1:
#                 Y[i,j]=(exp_phi_pred[i,j]/exp_phi_sum_vec[i])-1
#             else:
#                 Y[i,j]=(exp_phi_pred[i,j]/exp_phi_sum_vec[i])
#             Y[i,j+m]=-(exp_phi_pred[i,j]*(exp_phi_pred[i,j]-exp_phi_sum_vec[i])/
#                                              (exp_phi_sum_vec[i]*exp_phi_sum_vec[i]))
#     return np.asarray(Y)


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.nonecheck(False)
# @cython.cdivision(True)
# def c_str_entropy_link_der_12_vec(np.ndarray[dtype_int64_t, ndim=2] y_true,
#                                  double[:,:] exp_phi_pred,
#                                  double[:] exp_phi_sum_vec, 
#                                  double[:] weight_vec, 
#                                  dtype_int64_t[:,:,:] rp_tensor):
#     cdef long N = y_true.shape[0]
#     cdef long m = exp_phi_pred.shape[1]
#     cdef long ind = 0
#     cdef long qqq, xyz
#     cdef long yval = 0
#     cdef double curr_wt
#     cdef double[:,:] Y = np.zeros((N,2*m))
#     cdef double all_sum = 0
#     cdef double set_sum = 0
#     cdef long num_part = rp_tensor.shape[0]
#     cdef long i,j,k,t
#     cdef double jt, jk

#     for qqq in range(num_part):
#         curr_wt = weight_vec[ind]
        
#         for i in range(N):
#             xyz=0
#             # get y_true value from y_true onehot (passed in)
#             yval = 0
#             while(y_true[i,yval] == 0):
#                 yval+=1
#             while (rp_tensor[qqq,xyz,yval]==0):
#                 xyz+=1
#             set_sum = 0
#             for t in range(m):
#                 if (rp_tensor[qqq,xyz,t]==1):
#                     set_sum+=exp_phi_pred[i,t]
#             for j in range(m):
#                 # if j in set containing y_true
#                 jt = (exp_phi_pred[i,j]/set_sum)
#                 jk = (exp_phi_pred[i,j]/exp_phi_sum_vec[i])
#                 if (rp_tensor[qqq,xyz,j]==1):
#                     #Y[i,j]+= value when j in T * weight
#                     Y[i,j]+=(-curr_wt * (exp_phi_pred[i,j]*(exp_phi_sum_vec[i]-set_sum)/
#                                         (set_sum*exp_phi_sum_vec[i])))
#                     Y[i,j+m]+=(curr_wt * (jt-jk+jk*jk-jt*jt))
#                 else:
#                     #Y[i,j]+= value when j not in T * weight
#                     Y[i,j]+=curr_wt * exp_phi_pred[i,j]/exp_phi_sum_vec[i]
#                     Y[i,j+m]+=(-curr_wt * (-jk+jk*jk))
#     return np.asarray(Y)

