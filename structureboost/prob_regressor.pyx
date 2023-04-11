# cython: profile=True
# cython: language_level=3

import warnings
from prob_regr_unit import ProbRegressorUnit
from pdf_discrete import average_densities
from pdf_group import PdfGroup
import numpy as np
import pandas as pd
import ml_insights as mli
from joblib import Parallel, delayed


class PrestoBoost(object):
    """Nonparametric Probabilistic Regression using Coarse Learners.

    This is to be used with a numerical target when you wish to have a
    full predicted probability density conditional on the features.

        Parameters
    ----------

    num_forests : integer, greater than zero
        The number of forests to build.  Each forest fits a multiclass
        model on a quantized version of the target and converts the
        prediction to a probability density.

    num_trees : integer, or list-like
        The number of trees to use in each forest.  If a list, this can
        specify different numbers of trees for each forest.  If used in
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

    binpt_sample_size: int, default is 20
        The number of points to use in creating intervals for the coarse
        learners.  Ignored if the `binpt_method` is 'fixed'.

    binpt_vec : list-like, default is None
        The set of binpts to use for the 'fixed' or 'fixed-rss' methods
        of bin point selection.  Ignored if `binpt_method` is 'auto'.

    bin_interp : str, Default is'runif'
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

    structure_strides : str, Default is 'auto'
        Whether to use a structured entropy loss function or not.  Default is
        'auto', meaning to use the structured entropy and use the (rounded)
        square root of the number of intervals as the strides values.  The
        alternative is 'none' which will employ the standard cross-entropy loss.

    singleton_weight : float, default is 0.5
        When using structured entropy loss, how much to weight the singleton 
        partition.  Setting this equal to 1 reverts to the cross-entropy loss.
        Default is 0.5.

    max_depth : int, default is 3
        The maximum depth to use when building trees for the coarse classifier.\
        Smaller values are more likely to underfit and 
        larger values more likely to overfit.

    learning_rate : float, default is .05
        The "step size" to take when adding each new tree to the model.
        We recommend using a relatively low learning rate combined with
        early stopping.

    Other params are as in the main StructureBoost object and apply to the
    coarse learners.


    References
    ----------

    Lucena, B. "Nonparametric Probabilistic Regression with Coarse Learners"
    https://arxiv.org/abs/2210.16247
"""
    def __init__(self, num_forests, num_trees,  
                 feature_configs=None, binpt_method='auto',
                 binpt_sample_size=20,
                 binpt_vec=None,
                 bin_interp='runif',
                 structure_strides='auto',
                 singleton_weight=.5,
                 subsample=1,
                 range_extension='quantile_frac',
                 range_ext_param=(.25,.75,.25),
                 replace=True, min_size_split=25, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.05,
                 random_seed=0, na_unseen_action='weighted_random',
                 prec_digits=6, default_configs=None):
        self.num_forests = num_forests
        self.num_trees = num_trees
        self.binpt_sample_size = binpt_sample_size
        self.feature_configs = feature_configs
        self.binpt_method = binpt_method
        self.bin_interp = bin_interp
        self.binpt_vec = binpt_vec
        self.structure_strides=structure_strides
        self.singleton_weight=singleton_weight
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
        self.na_unseen_action=na_unseen_action
        if self.binpt_method=='fixed':
            self.strides_lp = int(np.round(np.sqrt(len(self.binpt_vec))))
            self.strides_hp = int(np.round(np.sqrt(len(self.binpt_vec))))
        else:
            self.strides_lp = int(np.round(np.sqrt(self.binpt_sample_size)))
            self.strides_hp = int(np.round(np.sqrt(self.binpt_sample_size)))
        self.range_extension = range_extension
        self.range_ext_param = range_ext_param
        self.calibrator_list = None
        self.prec_digits = prec_digits
        self.default_configs = default_configs
        if self.binpt_method=='fixed':
            self.range_extension='none'

        # figure out method (here or in fit)
        
    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
            early_stop_past_steps=1, choose_best_eval=True, fit_calibrators=False,
            refit_train_val=False, binpt_vec_list=None, calib_pw=100,
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

        fit_calibrators : bool, default is False
            When training with an eval_set, this will fit spline calibrators
            for each forest, that can then be optionally used at prediction
            time.  This is still fairly experimental.
"""
        self.forest_list = []
        self.analyze_y_train(y_train)
        if fit_calibrators:
            self.calibrator_list=[]
            for i in range(self.num_forests):
                self.calibrator_list.append(mli.SplineCalib(unity_prior=True, 
                    unity_prior_weight=calib_pw))


        # Determine the binpt_vecs for all forests
        if binpt_vec_list is None: # base case, we compute it
            self.binpt_vec_list=[]
            for i in range(self.num_forests):
                curr_binpt_vec = self.get_bin_pts(y_train, self.random_seed+i)
                self.binpt_vec_list.append(curr_binpt_vec.copy())
        else:   # Special case - we are given a binpt vec list
            self.binpt_vec_list = binpt_vec_list

        # Determine the number of trees for each forest
        if type(self.num_trees)==int:
            self.num_trees_of_forest = [self.num_trees]*self.num_forests
        else:
            # assume it is a list or array of ints
            self.num_trees_of_forest = self.num_trees

        for i in range(self.num_forests):
            # Set up forest with appropriate binpt_vec, number of trees
            self.forest_list.append(ProbRegressorUnit(
                                num_trees = self.num_trees_of_forest[i],
                                bin_vec = self.binpt_vec_list[i].copy(), 
                                feature_configs=self.feature_configs,
                                learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                structure_strides=self.structure_strides,
                                random_seed = self.random_seed+i,
                                sw=self.singleton_weight,
                                lp=self.strides_lp, hp=self.strides_hp,
                                prec_digits=self.prec_digits, 
                                default_configs=self.default_configs))


        if n_jobs==1: # if not parallelizing
            # Fit as appropriate using early stopping or not
            for i in range(self.num_forests):
                print(f'Training Forest {i}')
                if eval_set is None:
                    self.forest_list[i].fit(X_train, y_train, eval_freq=eval_freq)
                else:
                    self.forest_list[i].fit(X_train, y_train, eval_set, eval_freq,
                        early_stop_past_steps, choose_best_eval)

                    # fit calibrators if desired
                    if fit_calibrators:
                        X_valid = eval_set[0]
                        y_valid = eval_set[1]
                        validdiscpred = self.forest_list[i].gbtmodel.predict_proba(X_valid)
                        y_valid_bin = np.digitize(y_valid,self.forest_list[i].bin_vec[:-1]) -1
                        print('Fitting Calibrator')
                        self.calibrator_list[i].fit(validdiscpred,y_valid_bin)
        
            # If you want to refit on the combined training and validation data
            if refit_train_val and (eval_set is not None):
                num_trees_used = [self.forest_list[k].gbtmodel.num_trees_for_prediction 
                                    for k in range(self.num_forests)]
                X_valid = eval_set[0]
                y_valid = eval_set[1]
                X_train_val = pd.concat((X_train, X_valid))
                y_train_val = np.concatenate((y_train, y_valid))
                self.num_trees = num_trees_used
#                self.fit(X_train_val, y_train_val, binpt_vec_list=self.binpt_vec_list)

                num_trees_used = [self.forest_list[k].gbtmodel.num_trees_for_prediction 
                                    for k in range(self.num_forests)]
                self.forest_list = []
                for i in range(self.num_forests):
                    # Set up forest with appropriate binpt_vec, newnumber of trees
                    self.forest_list.append(ProbRegressorUnit(
                                        num_trees = num_trees_used[i],
                                        bin_vec = self.binpt_vec_list[i].copy(), 
                                        feature_configs=self.feature_configs,
                                        learning_rate=self.learning_rate,
                                        max_depth=self.max_depth,
                                        structure_strides=self.structure_strides,
                                        random_seed = self.random_seed+i,
                                        sw=self.singleton_weight,
                                        lp=self.strides_lp, hp=self.strides_hp))
                for i in range(self.num_forests):
                    print(f'Training Forest {i}')
                    self.forest_list[i].fit(X_train_val, y_train_val, eval_freq=eval_freq)



        else: # if parallelizing
            if eval_set is None:
                print('Training Forests')
                self.forest_list = Parallel(n_jobs=n_jobs,
                                        )(delayed(fit_and_return)(f, X_train, y_train) 
                                        for f in self.forest_list)
            else:
                print('Training Forests')
                self.forest_list = Parallel(n_jobs=n_jobs,
                                        )(delayed(fit_and_return2)(f, X_train, y_train, 
                                            eval_set, eval_freq,
                                            early_stop_past_steps, choose_best_eval) 
                                        for f in self.forest_list)

                if fit_calibrators:
                    print('Fitting Calibrators')
                    X_calib = eval_set[0]
                    y_calib = eval_set[1]
                    self.calibrator_list = Parallel(n_jobs=n_jobs,
                                        )(delayed(calib_parallel)(self.calibrator_list[i],
                                        self.forest_list[i].bin_vec, self.forest_list[i].gbtmodel,
                                         X_calib, y_calib) 
                                        for i in range(len(self.forest_list)))

                if refit_train_val:
                    print('Training Forests on combined data')
                    num_trees_used = [self.forest_list[k].gbtmodel.num_trees_for_prediction 
                                        for k in range(self.num_forests)]
                    self.forest_list = []
                    for i in range(self.num_forests):
                        # Set up forest with appropriate binpt_vec, newnumber of trees
                        self.forest_list.append(ProbRegressorUnit(
                                            num_trees = num_trees_used[i],
                                            bin_vec = self.binpt_vec_list[i].copy(), 
                                            feature_configs=self.feature_configs,
                                            learning_rate=self.learning_rate,
                                            max_depth=self.max_depth,
                                            structure_strides=self.structure_strides,
                                            random_seed = self.random_seed+i,
                                            sw=self.singleton_weight,
                                            lp=self.strides_lp, hp=self.strides_hp))

                    X_valid = eval_set[0]
                    y_valid = eval_set[1]
                    X_train_val = pd.concat((X_train, X_valid))
                    y_train_val = np.concatenate((y_train, y_valid))
                    self.forest_list = Parallel(n_jobs=n_jobs,
                                            )(delayed(fit_and_return)(f, X_train_val, y_train_val) 
                                            for f in self.forest_list)

    def analyze_y_train(self, y_train):
            self.unique_y = np.unique(y_train)
            self.num_uv = len(self.unique_y)
            self.y_max_train = np.max(y_train)
            self.y_min_train = np.min(y_train)


    def get_bin_pts(self, y_train, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        if self.binpt_method == 'fixed':
            binvec = np.array(self.binpt_vec)
        else:
            if (self.binpt_method=='fixed_rss'):
                binvec_start = np.array(self.binpt_vec)

                binvec = np.random.choice(binvec_start, replace=False, 
                       size=np.min([self.binpt_sample_size, len(binvec_start)]))
            elif (self.binpt_method=='auto') & (self.binpt_sample_size<self.num_uv):
                binvec = np.unique(np.quantile(y_train, np.random.uniform(size=self.binpt_sample_size)))
            else:
                binvec = self.unique_y
            if self.bin_interp == 'midpt':
                binvec = np.unique(
                            np.concatenate(([self.y_min_train],(binvec[1:]+binvec[:-1])/2, 
                                            [self.y_max_train])
                                          ))
            elif self.bin_interp == 'runif':
                binvec = np.unique(
                            np.concatenate(([self.y_min_train], 
                                         binvec[:-1]+(binvec[1:]-binvec[:-1])*np.random.uniform(size=len(binvec)-1), 
                                         [self.y_max_train])
                                          ))
            if (self.range_extension=='quantile_frac') and (self.binpt_method!='fixed_rss'):
                q_left, q_right, mult_factor = self.range_ext_param
                q_range = (np.quantile(y_train, q_right) - np.quantile(y_train, q_left))
                ext_amount = mult_factor*q_range
                added_pts = [self.y_min_train - ext_amount, self.y_max_train+ext_amount]
                #print(f'extending range: adding in {added_pts}')
                binvec = np.unique(np.concatenate((added_pts, binvec)))
        return(binvec)

    def predict_distributions(self, X_test, num_forests_to_use=-1, use_calibrators=False,
        calibrator_list=None, scaling='log'):
        if num_forests_to_use==-1:
            num_forests_to_use = self.num_forests
        elif (num_forests_to_use > self.num_forests):
            warnings.warn(f"Using only the {self.num_forests} that are available")
            num_forests_to_use = self.num_forests
        predlist = []
        for j in range(num_forests_to_use):
            if use_calibrators:
                if (calibrator_list is None):
                    if (self.calibrator_list is None):
                        warnings.warn("Calibrators not trained - aborting prediction")
                    else:
                        predlist.append(self.forest_list[j].predict_distributions(X_test, 
                            calibrator = self.calibrator_list[j]))
                else:
                    predlist.append(self.forest_list[j].predict_distributions(X_test, 
                        calibrator = calibrator_list[j]))

            else:
                predlist.append(self.forest_list[j].predict_distributions(X_test))
        final_dists = [average_densities([predlist[i][j] for i in range(num_forests_to_use)],
                                            scaling=scaling)
                                           for j in range(X_test.shape[0])]
        bv = final_dists[0].binvec
        pm = np.array([pdf.probvec for pdf in final_dists])
        return(PdfGroup(bv, pm))

def fit_and_return(model, X_train, y_train):
    model.fit(X_train, y_train, verbose=0)
    return(model)


def fit_and_return2(model, X_train, y_train, eval_set, eval_freq,
                        early_stop_past_steps, choose_best_eval):
    model.fit(X_train, y_train, eval_set, eval_freq,
                        early_stop_past_steps, choose_best_eval, verbose=0)
    return(model)

def calib_parallel(calibrator,binvec, model,X_calib, y_calib):
    validdiscpred = model.predict_proba(X_calib)
    y_calib_bin = np.digitize(y_calib,binvec[:-1]) -1
    calibrator.fit(validdiscpred,y_calib_bin)
    return(calibrator)

