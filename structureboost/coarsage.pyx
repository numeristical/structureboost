# cython: profile=True
# cython: language_level=3

from structure_gb_multi import StructureBoostMulti, get_one_hot_mat
from structure_gb_multi import c_str_entropy_link_der_12_vec_sp, c_entropy_link_der_12_vec_sp
from structure_dt_multi import StructureDecisionTreeMulti
from pdf_discrete import PdfDiscrete, get_part, chain_partition, get_pdf_from_data
from pdf_group import PdfGroup, log_loss_pdf
from structure_gb import get_basic_config, default_config_dict
import numpy as np
import pandas as pd
import warnings

cimport numpy as np
cimport cython
from libc.math cimport log as clog


class Coarsage(StructureBoostMulti):
    """Coarse Adjustment Boosting for Nonparametric Probabilistic Regression.

    This method is to be used with a numerical target when you wish to have a
    full predicted probability density conditional on the features.

    Parameters
    ----------

    num_trees : integer
        The number of trees to use in each forest. If used in
        conjunction with early stopping, this will be the maximum number
        of trees considered.

    feature_configs : dict
        A dictionary that contains the specifications for each feature in
        the models (as in a standard StructureBoost model).  StructureBoost
        allows an unusual amount of flexibility
        which necessitates a more complicated configuration. See
        https://structureboost.readthedocs.io/en/latest/feat_config.html
        for details.

    binpt_method : str, default is 'auto'
        How to choose the intervals (bin endpoints) for the coarse learners.
        Default is 'auto', which means a random set of quantiles (of size
        binpt_sample_size) is chosen from the training data, and the interval 
        endpoints are chosen between those values (using the method indicated
        by `bin_interp`).  Other options are `fixed` (a given set of intervals
        is used for each forest) or `fixed-rss` (a random sample of a fixed
        set of endpoints is used)

    binpt_vec : list-like, default is None
        The set of binpts to use for the 'fixed' or 'fixed-rss' methods
        of bin point selection.  Ignored if `binpt_method` is 'auto'.

    num_coarse_bins: int, default is 40
        The number of points to use in creating intervals for the coarse
        learners.  Ignored if the `binpt_method` is 'fixed'.

    structured_loss : bool, Default is True
        Whether or not to use a structured entropy loss function. The
        structured entropy loss helps regularize by recognizing that the
        numerical target has an ordinal structure.  It is configured with
        two parameters, the 'singleton_weight' and the 'structure_block_size'.

    singleton_weight : float, default is 0.5
        When using structured entropy loss, how much to weight the singleton 
        partition.  Numbers closer to 1 give less "partial credit" for being close.
        Setting this equal to 1 reverts to the standard cross-entropy loss. This
        parameter is ignored if `structured_loss` is False. Default is 0.5.

    structure_block_size : int, default is None
        When using structured entropy loss, this determines how far away the
        "partial credit" kicks in (similar to a kernel width).  Smaller values
        will behave closer to the standard cross-entropy and larger values
        will begin to give "partial credit" further away.  The min value is 2
        and the maximum value is the number of bins-1.  Default is None, which
        means the algorithm will choose a value automatically. Currently this
        is done by taking the (rounded) square root of the number of bins.

    bin_interp : str, Default is 'runif'
        How to create the bin points from the randomly chosen target values
        when `binpt_method` is 'auto'.  Default is 'runif' which means we
        draw uniformly in the space between neighboring values.  Option 'midpt'
        means you simply choose the midpoint between the two neighboring values.

    range_extension : str, default is 'quantile_frac'
        How (and if) to extend the range of support beyond the values seen in the 
        training data.  Considered only if `binpt_method` is 'auto'.  Default
        is `quantile_frac` which means we find the size of the quantile range 
        determined by the first two values in `range_ext_param`, multiply that
        value by the third given number, and add a bin of that size to either end.

    range_ext_param : tuple/list of float, default is (.25,.75,.25)
        Three values used to determine how large the additional bins should
        be on each side.  Default is .25, .75, .25 which means that the IQR
        (inter-quartile range) of the training data is computed, and its
        width is multiplied by .25.

    max_depth : int, default is 3
        The maximum depth to use when building trees for the coarse classifier.\
        Smaller values are more likely to underfit and 
        larger values more likely to overfit.

    learning_rate : float, default is .05
        The "step size" to take when adding each new tree to the model.
        We recommend using a relatively low learning rate combined with
        early stopping.

    binpt_growth_wait : int, default is 4
        How many iterations to wait between "growth" steps (where you sample
        potentially new bin pts).  A setting of 0 means you sample new points
        every step.  Default is 4 which means it grows every fifth step. The
        purpose is to avoid growing the resolution too quickly, which can slow
        down the training.

    initial_growth_steps : int, default is 2
        The number of growth steps to take initially (regardless of the setting
        of bin_growth_wait). The default is 2.  It is recommended to keep this
        parameter at least 2.

    Other params are as in the main StructureBoost object and apply to the
    coarse learners.


    References
    ----------

    Lucena, B. To appear
    """

    def __init__(self, num_trees,
                 max_depth,
                 feature_configs=None,
                 binpt_method='auto',
                 binpt_vec=None,
                 num_coarse_bins=40,
                 structured_loss=True,
                 structure_block_size=None,
                 singleton_weight=.4,
                 max_resolution=3000,
                 bin_interp='runif',
                 range_extension='quantile_frac',
                 range_ext_param=(.25,.75,.25),
                 learning_rate=.02,
                 binpt_growth_wait=0,
                 initial_growth_steps=2,
                 replace=True, min_size_split=25,
                 subsample=1,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1,
                 random_seed=0, initial_model=None,
                 na_unseen_action='weighted_random',
                 prec_digits=6, default_configs=None):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.num_coarse_bins = num_coarse_bins
        self.feature_configs = feature_configs
        self.binpt_method = binpt_method
        self.bin_interp = bin_interp
        self.binpt_vec = binpt_vec
        self.structured_loss=structured_loss
        self.structure_block_size=structure_block_size
        self.singleton_weight=singleton_weight
        self.max_resolution=max_resolution
        self.binpt_growth_wait=binpt_growth_wait
        self.initial_growth_steps = initial_growth_steps
        self.subsample=subsample
        self.replace=replace
        self.min_size_split=min_size_split
        self.max_depth=max_depth
        self.gamma=gamma
        self.reg_lambda=reg_lambda
        self.feat_sample_by_tree=feat_sample_by_tree
        self.feat_sample_by_node=feat_sample_by_node
        self.learning_rate=learning_rate
        self.random_seed=random_seed
        self.initial_model=initial_model
        self.na_unseen_action=na_unseen_action
        if self.binpt_method=='fixed_rss':
            self.binpt_method='fixed-rss'
        if self.binpt_method not in ['fixed-rss', 'fixed','auto']:
            raise ValueError('unknown binpt_method')
        if self.structure_block_size is None:
            if self.binpt_method=='fixed':
                self.structure_block_size = int(np.round((len(self.binpt_vec)-1)/4))
            else:
                self.structure_block_size = int(np.round((self.num_coarse_bins)/4))

        if self.structure_block_size>=self.num_coarse_bins:
            raise ValueError("structure_block_size must be strictly < num_coarse_bins")
        self.range_extension = range_extension
        if self.binpt_method=='fixed-rss':
            self.range_extension='none'
        self.range_ext_param = range_ext_param
        if self.binpt_method=='fixed':
            self.range_extension='none'
        if feature_configs is not None:
            self.feature_configs = feature_configs.copy()
            self._process_feature_configs()
            self.feat_list_full = list(self.feature_configs.keys())
        else:
            self.feature_configs=None
        self.prec_digits = prec_digits
        if default_configs is None:
            self.default_configs = default_config_dict()

    def analyze_y_train(self, y_train):
            self.unique_y = np.unique(y_train)
            self.num_uv = len(self.unique_y)
            self.y_max_train = np.max(y_train)
            self.y_min_train = np.min(y_train)

    def _process_y_data(self, y_data):
        if type(y_data) == pd.Series:
            y_data = y_data.to_numpy().astype(float)
        elif type(y_data) == np.ndarray:
            y_data = y_data.astype(float)
        return(y_data)

    def _get_initial_pred(self, X_train, y_train):
        if self.initial_model is not None:
            return(self.initial_model.predict_distributions(X_train))
        else:
            init_bin_pts = self.get_bin_pts(y_train, random_seed=2**32-1-self.random_seed)
            self.init_pdf = get_pdf_from_data(init_bin_pts, y_train)
            return(PdfGroup(binvec = self.init_pdf.binvec,
                probmat = np.tile(self.init_pdf.probvec, (X_train.shape[0],1))))

    def _update_curr_answer(self, curr_answer, X_train, index):
        if (index>0):
            delta_mat = self.dec_tree_list[index-1].predict(X_train)
            curr_answer = self._update_pdfs(curr_answer, delta_mat,
                                            self.binpt_vec_list[index-1], index)
        return(curr_answer)

    def _update_pdfs(self, answer, delta_mat, delta_binpt_vec, index):
        density_binpts = answer.binvec.copy()
        no_new_binpts = np.all(np.isin(delta_binpt_vec, density_binpts))
        fine_binpts = np.unique(np.concatenate((density_binpts, delta_binpt_vec))) 
        fine_bin_widths = np.diff(fine_binpts)
        num_rows = answer.densitymat.shape[0]
        num_fine_bins = len(fine_binpts)-1
        bigmat = update_densities_to_logprobs(answer.densitymat, 
                               density_binpts, delta_mat, delta_binpt_vec,
                               fine_binpts, fine_bin_widths,
                               self.learning_rate)
        bigmat = self.softmax_mat(bigmat)
        if index>=2:
            answer.probmat=None
            answer.densitymat=None
            delta_mat=None
            fine_bin_widths=None
        return(PdfGroup(binvec=fine_binpts, probmat=bigmat))

    def process_curr_answer(self, curr_answer):
        return curr_answer

    def _compute_loss(self, y_true, pred):
        return(log_loss_pdf(y_true, pred))

    def compute_gh_mat(self, y_train, curr_answer, index):
        # Bin the y-values according to the bins for this specific tree
        curr_binpts = self.binpt_vec_list[index]
        curr_num_classes = len(curr_binpts)-1
        y_bin = (np.digitize(y_train, curr_binpts) -1 
                    -(y_train==curr_binpts[-1]))
        curr_answer_bin_probs = curr_answer.bins_to_probs(curr_binpts)
        if not self.structured_loss:
            self.ts_dict = None
            y_g_h_mat = c_entropy_link_der_12_vec_sp(y_bin, curr_answer_bin_probs,
                                                np.ones(len(y_train)))
            return y_g_h_mat
        else:
            self.stride_list = list(range(self.structure_block_size, self.structure_block_size+1))
            pl = []
            for i in self.stride_list:
                pl = pl+chain_partition(curr_num_classes,i)
            ts={}
            ts['partition_type']='fixed'
            ts['partition_list'] = pl
            ts['singleton_weight'] = self.singleton_weight
            ts['partition_weight_vec'] = np.ones(len(pl))*(1-self.singleton_weight)/len(pl)
            self.ts_dict=ts
            self.part_weight_vec = np.array(self.ts_dict['partition_weight_vec'])
            self.rpt = self._create_rpt_from_list(self.ts_dict['partition_list'], curr_num_classes)
            y_g_h_mat = c_str_entropy_link_der_12_vec_sp(y_bin, curr_answer_bin_probs,
                                                np.ones(len(y_train)),
                                        self.part_weight_vec, self.rpt)
            if self.singleton_weight>0:
                y_g_h_mat_reg = c_entropy_link_der_12_vec_sp(y_bin, curr_answer_bin_probs,
                                                np.ones(len(y_train)))
                y_g_h_mat = y_g_h_mat + self.singleton_weight*y_g_h_mat_reg
            return y_g_h_mat

    def _create_rpt_from_list(self, partition_list, num_classes):
        num_part = len(partition_list)
        max_part_size = np.max(np.array([len(qq) for qq in partition_list]))
        rpt = np.zeros((num_part, max_part_size, num_classes), dtype=np.int_)
        flat_list = [j for sl in partition_list for i in sl for j in i]
        min_val, max_val = np.min(flat_list), np.max(flat_list)
        if (min_val<0) or (max_val>num_classes-1):
            w_str = "Elements of partition must be >=0 and <=(num_classes-1). "
            w_str+= "Results may be unexpected."
            warnings.warn(w_str)
        for i in range(num_part):
            for j in range(len(partition_list[i])):
                for k in partition_list[i][j]:
                        rpt[i,j,k]=1
        return(rpt)

    def _add_train_next_tree(self, features_for_tree, X_train, y_g_h_train, index):
        self.dec_tree_list.append(StructureDecisionTreeMulti(
                        feature_configs=self.feature_configs,
                        feature_graphs=self.feature_graphs,
                        num_classes=self.num_classes[index],
                        min_size_split=self.min_size_split,
                        gamma=self.gamma, max_depth=self.max_depth,
                        reg_lambda=self.reg_lambda,
                        feat_sample_by_node=self.feat_sample_by_node))
        self.dec_tree_list[index].fit(X_train, y_g_h_train,
                                  feature_sublist=features_for_tree,
                                  uv_dict=self.unique_vals_dict)

    def get_tensors_for_predict(self):
        if self.optimizable:
            cat_size = np.max(np.array([dt.get_max_split_size() for dt in self.dec_tree_list]))
            num_dt = len(self.dec_tree_list)
            max_nodes = np.max(np.array([dt.num_nodes for dt in self.dec_tree_list]))
            max_num_classes = np.max(self.num_classes)
            self.pred_tens_int = np.zeros((num_dt, max_nodes, cat_size+6), dtype=np.int_)-1
            self.pred_tens_float = np.zeros((num_dt, max_nodes, max_num_classes+2))
            for i in range(num_dt):
                self.convert_dt_to_matrix(i)
            self.optimized=True
        else:
            print("Model not optimizable for predict due to string or voronoi variable.")

    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
        early_stop_past_steps=1, choose_best_eval=True, 
        refit_train_val=False, 
        n_jobs=1, verbose=1):
        """Fit the prob regressor given training features and target values.

        Parameters
        ----------

        X_train : DataFrame
            A dataframe containing the features.  StructureBoost uses the
            column names rather than position to locate the features.  The
            set of features is determined by the feature_configs *not* 
            X_train.  Consequently, it is OK to pass in extra columns that
            are not used by the `feature_configs`.

        y_train : Series or numpy array
            The numerical target values.  Must have length the same size as
            the number of rows in X_train. 

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

        refit_train_val : bool, default is False
            When training with an eval_set and early stopping, this will run
            two passes of training.  The first will calculate the optimal
            number of trees for each forest using early stopping on the
            eval_set. The second pass will refit on the combined training and
            validation (eval_set) data using the numbers of trees calculated
            in the first step
"""
        self.dec_tree_list = []
        self.num_classes = []
        self.analyze_y_train(y_train)
        if (self.range_extension=='quantile_frac') and (self.binpt_method!='fixed-rss'):
            q_left, q_right, mult_factor = self.range_ext_param
            q_range = (np.quantile(y_train, q_right) - np.quantile(y_train, q_left))
            self.ext_amount = mult_factor*q_range

        # Determine the binpt_vecs for all trees
        self.binpt_vec_list = []
        # bpvec = np.array([])
        ## Revisit this to catch errors and (corner) cases
        i = 0
        curr_res = 0
        # for i in range(lb):
        self.binpt_vec_final = np.array([])
        while (curr_res<(self.max_resolution - self.num_coarse_bins)) and (i<self.num_trees):
            ngm = (i % (self.binpt_growth_wait+1)) != 0 #grow only once every bgw+1 steps
            ngm = ngm and (i>=self.initial_growth_steps)
            curr_binpt_vec = self.get_bin_pts(y_train, self.random_seed+i, no_growth_mode=ngm)
            self.binpt_vec_list.append(curr_binpt_vec.copy())
            self.num_classes.append(len(curr_binpt_vec)-1)
            self.binpt_vec_final = np.unique(np.concatenate((self.binpt_vec_final, curr_binpt_vec)))
            curr_res = len(self.binpt_vec_final)-1
            i+=1
        # if self.binpt_method=='auto':
        #     self.binpt_vec_final = bpvec
        while (i<self.num_trees):
            curr_binpt_vec = self.get_bin_pts(y_train, self.random_seed+i, no_growth_mode=True)
            self.binpt_vec_list.append(curr_binpt_vec.copy())
            self.num_classes.append(len(curr_binpt_vec)-1)
            i+=1
        self.num_classes=np.array(self.num_classes, dtype=np.int_)

        super().fit(X_train, y_train, eval_set=eval_set, eval_freq=eval_freq,
            early_stop_past_steps=early_stop_past_steps, 
            choose_best_eval=choose_best_eval, verbose=verbose)


    def predict_distributions(self, X_test, num_trees_to_use=-1, same_col_pos=True):
        if (type(X_test)==np.ndarray):
            if self.optimizable:
                if self.optimized:
                    return(self._predict_dist_fast(X_test, num_trees_to_use))
                else:
                    self.get_tensors_for_predict()
                    return(self._predict_dist_fast(X_test, num_trees_to_use))
            else:
                # clunky.. make X_test into a pd.DataFrame with appr columns
                # but this would be an unusual situation
                inv_cti = {j:i for (i,j) in self.column_to_int_dict.items}
                col_list = [inv_cti[i] if i in inv_cti.keys() else '_X_'+str(i)
                            for i in X_test.shape[0]]
                return(self._predict_dist_py(pd.DataFrame(X_test, columns=col_list), num_trees_to_use))
        elif (type(X_test)==pd.DataFrame):
            if same_col_pos:
                if self.optimizable:
                    if not self.optimized:
                        self.get_tensors_for_predict()
                    return(self._predict_dist_fast(X_test.to_numpy(), num_trees_to_use))
                else:
                    return(self._predict_dist_py(X_test, num_trees_to_use))
            else:
                return(self._predict_dist_py(X_test, num_trees_to_use))

    def _predict_dist_fast(self, X_test, int num_trees_to_use=-1,eps=1e-16):

        if not self.optimized:
            if self.optimizable:
                self.get_tensors_for_predict()
            else:
                print('Fast predict not possible for this model type')
                return None

        cdef int i
        num_rows = X_test.shape[0]
        if num_trees_to_use == -1:
            num_trees_to_use = self.num_trees_for_prediction
        fine_binpts = self.get_fine_binpts(num_trees_to_use)
        if self.initial_model is None:
            init_pdf_fine = self.init_pdf.add_binpts(fine_binpts)
            init_logprobmat_fine = np.tile(np.log(np.maximum(
                                            init_pdf_fine.probvec,eps)), 
                                                (num_rows,1))
        else:
            warnings.warn("Initial model not yet implented")
        tree_pred_tens_coarse = predict_with_tensor_c_mc(
                                self.pred_tens_float[:num_trees_to_use,:,:],
                                self.pred_tens_int[:num_trees_to_use,:,:],
                                X_test, self.num_classes)
        bv_mat = np.array(self.binpt_vec_list).copy()
        out_mat = tensor_result_sum_fine(tree_pred_tens_coarse, bv_mat, fine_binpts,
                                        self.learning_rate)
        out_mat = init_logprobmat_fine + out_mat
        probmat = self.softmax_mat(out_mat)
        return(PdfGroup(fine_binpts, probmat=probmat))

    def get_fine_binpts(self, num_trees_to_use):
        fbp = np.concatenate([self.binpt_vec_list[i] 
                                    for i in range(num_trees_to_use)])
        fbp = np.unique(np.concatenate((fbp, self.initial_pred.binvec)))
        return(fbp)

    def get_pts_to_sample(self, y_train, no_growth_mode=False):
        if no_growth_mode:
            pts_to_sample = np.setdiff1d(np.array(self.binpt_vec_final), 
                                        [self.y_min_train,self.y_max_train,
                                        np.min(self.binpt_vec_final), np.max(self.binpt_vec_final)])

        elif self.binpt_method=='fixed-rss':
            pts_to_sample =  np.setdiff1d(np.array(self.binpt_vec), 
                                            [self.y_min_train,self.y_max_train,
                                            np.min(self.binpt_vec), np.max(self.binpt_vec)])
        elif self.binpt_method=='auto':
            num_sample_pts = self.num_coarse_bins-2 + 2*(self.range_extension=='none')
            if num_sample_pts>(self.num_uv-2):
                warnings.warn('Not enough unique values in y_train for num_coarse_bins')
            pts_to_sample = np.unique(np.quantile(y_train, np.random.uniform(size=num_sample_pts*2+10)))
            pts_to_sample = np.setdiff1d(pts_to_sample,[self.y_min_train,self.y_max_train])
            shortfall = len(pts_to_sample)-num_sample_pts
            if shortfall>0:
                newpt_cands = np.setdiff1d(self.unique_y, np.union1d(
                            pts_to_sample,[self.y_min_train,self.y_max_train]))
                newpts = np.random.choice(newpt_cands, shortfall)
                pts_to_sample = np.union1d(pts_to_sample, newpts)
        # pts_to_sample = np.sort(pts_to_sample) #shouldn't need this since we sort in get_bin_pts
        return(pts_to_sample)

    def get_bin_pts(self, y_train, random_seed=None, no_growth_mode=False):
        if random_seed is not None:
            np.random.seed(random_seed)
        if self.binpt_method == 'fixed':
            return(self.binpt_vec) # should I return a copy here?
        else:
            pts_to_sample = self.get_pts_to_sample(y_train, no_growth_mode)
            num_sample_pts = self.num_coarse_bins-2 + 2*(self.range_extension=='none')
            num_avail = len(pts_to_sample)
            if num_sample_pts<=num_avail:
                sampled_pts = np.random.choice(pts_to_sample, replace=False, size=num_sample_pts)
            else:
                warnings.warn(f'Trying to sample {num_sample_pts} pts from {num_avail} available')
            if no_growth_mode:
                binvec = np.unique(
                            np.concatenate(([self.y_min_train],sampled_pts[1:], 
                                            [self.y_max_train])
                                          ))
            elif self.binpt_method=='fixed-rss':
                binvec = np.unique(
                            np.concatenate(([np.min(self.binpt_vec)],sampled_pts[1:], 
                                            [np.max(self.binpt_vec)])
                                          )) 

            elif (self.bin_interp == 'midpt'):
                binvec = np.unique(
                            np.concatenate(([self.y_min_train],(sampled_pts[1:]+sampled_pts[:-1])/2, 
                                            [self.y_max_train])
                                          ))
            elif (self.bin_interp == 'runif'):
                binvec = np.unique(
                            np.concatenate(([self.y_min_train], 
                                         sampled_pts[:-1]+(sampled_pts[1:]-sampled_pts[:-1])*
                                         np.random.uniform(size=len(sampled_pts)-1), 
                                         [self.y_max_train])
                                          ))
            if (self.range_extension=='quantile_frac') and (self.binpt_method!='fixed-rss'):
                added_pts = [self.y_min_train - self.ext_amount, self.y_max_train+self.ext_amount]
                binvec = np.unique(np.concatenate((added_pts, binvec)))
            if len(binvec)!=(self.num_coarse_bins+1):
                warnings.warn("wrong number of bins in random creation of intervals")
        return(binvec)

    def convert_dt_to_matrix(self, dt_num):
        curr_node = self.dec_tree_list[dt_num].dec_tree
        self.convert_subtree(curr_node, dt_num)

    def convert_subtree(self, node, dt_num):
        ni = node['node_index']
        if node['node_type']=='leaf':
            self.pred_tens_int[dt_num, ni, 0]= 0
            self.pred_tens_float[dt_num, ni, 1] = float(node['num_data_points'])
            self.pred_tens_float[dt_num, ni, 2:self.num_classes[dt_num]+2] = node['node_summary_val']
        else:
            self.pred_tens_float[dt_num, ni, 1] = float(node['num_data_points'])
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


def manage_zeros(probvec):
    mpv = np.clip(probvec, 1e-16, 1-1e-16)
    mpv = mpv/np.sum(mpv)
    return(mpv)

def manage_zeros2(probvec):
    mpv = np.minimum(probvec, 1e-16)
    # mpv = mpv/np.sum(mpv)
    return(mpv)

def get_result_tensor(binvec_list, fine_binvec, coarse_result_tens):
    num_fine_int = len(fine_binvec)-1
    num_rows, num_trees = coarse_result_tens.shape[:2]
    fine_result_tens = np.zeros((num_rows, num_trees, num_fine_int))
    for i in range(num_trees):
        fine_result_tens[:,i,:] = expand_result_mat(binvec_list[i], fine_binvec,
                                                   len(binvec_list[i])-1, num_fine_int, num_rows,
                                                   coarse_result_tens[:,i,:], fine_result_tens[:,i,:])
    return fine_result_tens

@cython.boundscheck(False)
@cython.wraparound(False)
def expand_result_mat(np.ndarray[double] coarse_interval_pts,
                      np.ndarray[double] fine_interval_pts, 
                      long num_coarse_int, long num_fine_int, long num_rows,
                      np.ndarray[double, ndim=2] coarse_results,
                      np.ndarray[double, ndim=2] fine_results):
    # Assume that all the coarse interval pts are in fine interval pts
    cdef long coarse_ptr=0, fine_ptr=0 #we assume the first pts are equal in both int pts
    cdef long row_ptr=0

    while ((coarse_ptr+1)<=num_coarse_int):
        while (fine_ptr<num_fine_int) and (fine_interval_pts[fine_ptr+1] <= coarse_interval_pts[coarse_ptr+1]):
            for row_ptr in range(num_rows):
                fine_results[row_ptr,fine_ptr] = coarse_results[row_ptr,coarse_ptr]
            fine_ptr+=1
        coarse_ptr+=1
    return(fine_results)

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_with_tensor_c_mc(np.ndarray[double, ndim=3] dtm_float,
                      np.ndarray[long, ndim=3] dtm,
                      np.ndarray[double, ndim=2] feat_array,
                      np.ndarray[long] num_classes_arr):
    """This is the same as the structure_gb_multi version except for num_classes.

    Specifically, this permits that different trees may have a different 
    number of classes."""
    cdef long cat_vals_end
    cdef long max_num_classes = np.max(num_classes_arr)
    cdef long num_classes
    cdef np.ndarray[double, ndim=3] res_tens = np.zeros((
                                feat_array.shape[0], dtm.shape[0], max_num_classes))
    cdef long cn, ri, ind, j, k, q
    cdef double curr_val, ind_doub
    cdef bint at_leaf, found_val
    cdef np.ndarray[long, ndim=2] isnan_array = np.isnan(feat_array).astype(int)
    
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

    # meanings of values for NODE_TYPE in dtm_int
    cdef long LEAF = 0
    cdef long NUMER = 1
    cdef long CATEG = 2

    for k in range(dtm.shape[0]):
        num_classes=num_classes_arr[k]
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
def update_densities_to_logprobs(np.ndarray[double, ndim=2] densitymat, 
                               np.ndarray[double] density_binpts, 
                               np.ndarray[double, ndim=2] deltamat, 
                               np.ndarray[double] delta_binpts,
                               np.ndarray[double] fine_binpts,
                               np.ndarray[double] fine_bin_widths,
                               double lr):
    cdef long num_rows = densitymat.shape[0]
    cdef long num_fine_bins = len(fine_binpts)-1
    cdef double eps=1e-16
    cdef long dens_ptr =0
    cdef long delta_ptr = 0
    cdef long row_ptr, fine_ptr
    cdef double fine_prob, fbw, curr_delta
    cdef np.ndarray[double, ndim=2] fine_logprobmat = np.zeros((num_rows, num_fine_bins))
    
    for fine_ptr in range(num_fine_bins):
        fbw = fine_bin_widths[fine_ptr]
        for row_ptr in range(num_rows):
            fine_prob = fbw*densitymat[row_ptr, dens_ptr]
            curr_delta = deltamat[row_ptr,delta_ptr]
            if fine_prob<eps:
                fine_prob=eps
            fine_logprobmat[row_ptr, fine_ptr] = clog(fine_prob) + lr*curr_delta
        if fine_binpts[fine_ptr+1] >= density_binpts[dens_ptr+1]:
            dens_ptr+=1
        if fine_binpts[fine_ptr+1] >= delta_binpts[delta_ptr+1]:
            delta_ptr+=1
    return(fine_logprobmat)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def tensor_result_sum_fine(np.ndarray[double, ndim=3] coarse_pred,
                           np.ndarray[double, ndim=2] bv_mat,
                           np.ndarray[double] fine_binpts, double lr):

    cdef long num_rows = coarse_pred.shape[0]
    cdef long nt = coarse_pred.shape[1]
    cdef long num_fi = fine_binpts.shape[0]-1
    cdef np.ndarray[double, ndim=2] outmat = np.zeros((num_rows,num_fi))
    cdef long fi_ptr = 0
    cdef np.ndarray[long] coarse_ptr_arr = np.zeros(nt, dtype=np.int_)
    cdef long row_ptr = 0

    for fi_ptr in range(num_fi):
        for i in range(nt):
            for row_ptr in range(num_rows):
                outmat[row_ptr, fi_ptr] += coarse_pred[row_ptr, i, coarse_ptr_arr[i]] * lr
            if fine_binpts[fi_ptr+1] >= bv_mat[i,coarse_ptr_arr[i]+1]:
                coarse_ptr_arr[i]+=1
    return(outmat)

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# def update_densities_to_logprobs_simple(np.ndarray[double, ndim=2] probmat, 
#                                np.ndarray[double] fine_binpts,
#                                np.ndarray[double, ndim=2] deltamat, 
#                                np.ndarray[double] delta_binpts,
#                                double lr):
    
#     cdef long num_rows = probmat.shape[0]
#     cdef long num_fine_bins = len(fine_binpts)-1
#     cdef double eps=1e-16
#     cdef long delta_ptr = 0
#     cdef long row_ptr, fine_ptr
#     cdef double fine_prob, curr_delta
#     cdef np.ndarray[double, ndim=2] fine_logprobmat = np.zeros((num_rows, num_fine_bins))
    
#     for fine_ptr in range(num_fine_bins):
#         for row_ptr in range(num_rows):
#             fine_prob = probmat[row_ptr, fine_ptr]
#             curr_delta = deltamat[row_ptr,delta_ptr]
#             if fine_prob<eps:
#                 fine_prob=eps
#             fine_logprobmat[row_ptr, fine_ptr] = clog(fine_prob) + lr*curr_delta
#         fine_ptr+=1
#         if fine_binpts[fine_ptr+1] >= delta_binpts[delta_ptr+1]:
#             delta_ptr+=1
#     return(fine_logprobmat)

