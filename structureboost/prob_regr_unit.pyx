# cython: profile=True
# cython: language_level=3

from structure_gb_multi import StructureBoostMulti
from pdf_discrete import PdfDiscrete, get_part, chain_partition, density_plot
import numpy as np
import pandas as pd
import warnings

class ProbRegressorUnit(object):
    
    def __init__(self, num_trees, feature_configs,
                 bin_vec, structure_strides='auto',
                 subsample=1,
                 replace=True, min_size_split=25, max_depth=3,
                 gamma=0, reg_lambda=1, feat_sample_by_tree=1,
                 feat_sample_by_node=1, learning_rate=.02,
                 random_seed=0, na_unseen_action='weighted_random', sw=.01, lp=2, hp=5):
        self.bin_vec = bin_vec
        self.num_classes = len(self.bin_vec)-1
        if (type(structure_strides)==str) and (structure_strides=='auto'):
            self.structure_strides = list(range(lp, hp+1))
            pl = []
            for i in self.structure_strides:
                pl = pl+chain_partition(self.num_classes,i)
            ts={}
            ts['partition_type']='fixed'
            ts['partition_list'] = pl
            ts['singleton_weight'] = sw
            ts['partition_weight_vec'] = np.ones(len(pl))*(1-sw)/len(pl)
        else:
            ts = None
        self.gbtmodel = StructureBoostMulti(
                            num_trees=num_trees,
                            feature_configs=feature_configs,
                            num_classes=self.num_classes,
                            target_structure=ts,
                            subsample=subsample,
                            replace=replace,
                            min_size_split=min_size_split,
                            max_depth=max_depth,
                            gamma=gamma, reg_lambda=reg_lambda,
                            feat_sample_by_tree=feat_sample_by_tree,
                            feat_sample_by_node=feat_sample_by_node,
                            learning_rate=learning_rate,
                            random_seed=random_seed,
                            na_unseen_action=na_unseen_action)

    def fit(self, X_train, y_train, eval_set=None, eval_freq=10,
            early_stop_past_steps=1, choose_best_eval=True, verbose=1):
        y_bin = np.digitize(y_train,self.bin_vec[:-1]) -1
        if np.min(y_bin)<0:
            warnings.warn('Target training values must be inside of range designated by bin_vec')
        if eval_set is not None:
            y_bin_eval = np.digitize(eval_set[1],self.bin_vec[:-1]) -1
            self.gbtmodel.fit(X_train, y_bin, eval_set=(eval_set[0],y_bin_eval), 
                                eval_freq=eval_freq,
                                early_stop_past_steps=early_stop_past_steps,
                                choose_best_eval=choose_best_eval,
                                verbose=verbose)
        else:
            self.gbtmodel.fit(X_train, y_bin, eval_freq=eval_freq, verbose=verbose)

    def predict_distributions(self, X_test, num_trees_to_use=-1, same_col_pos=True,
        calibrator=None):
        if calibrator is None:
            prob_mat = self.gbtmodel.predict(X_test, num_trees_to_use)
        else:
            prob_mat = calibrator.calibrate(self.gbtmodel.predict(X_test, num_trees_to_use))
        self.prob_mat = prob_mat
        self.pred_dists = [PdfDiscrete(self.bin_vec, prob_mat[i,:]) for i in range(prob_mat.shape[0])]
        return self.pred_dists