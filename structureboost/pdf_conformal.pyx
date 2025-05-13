# cython: profile=True
# cython: language_level=3

import warnings
from pdf_discrete import PdfDiscrete
from pdf_group import PdfGroup
from pdf_set import PdfSet
import numpy as np
import pandas as pd
import ml_insights as mli

class PdfConformalizer(object):

    def __init__(self, method='reweight'):
        self.method=method

    def fit(self, dists_calib, y_calib, depth):
        self.adj_dict={}
        self.depth=depth
        triplet_array = make_triplet_array(depth)
        if type(dists_calib)==PdfGroup:
            if dists_calib.pdf_list is None:
                dists_calib.make_pdf_list
            dists_calib = dists_calib.pdf_list
        next_dl_calib = [dists_calib[i] for i in range(len(y_calib))]
        for j in range(triplet_array.shape[0]):
            q_left, q_ctr, q_right = triplet_array[j,:]
            print(f'calibrating {q_ctr} with boundaries {q_left, q_right}')
            resids = y_calib - np.array([next_dl_calib[i].quantile(q_ctr) for i in range(len(y_calib))])
            adj = np.quantile(resids, q_ctr)
            self.adj_dict[(q_ctr, q_left, q_right)]=adj
            print(f'adjustment is {adj}')
            next_dl_calib = [conformalize_step_dist(next_dl_calib[i], q_ctr, q_left,q_right, adj) 
                       for i in range(len(y_calib))]

    def calibrate(self, dists):
        triplet_array = make_triplet_array(self.depth)
        if type(dists)==PdfGroup:
            if dists.pdf_list is None:
                dists.make_pdf_list
            dists = dists.pdf_list
        next_dl_out = [dists[i] for i in range(len(dists))]
        for j in range(triplet_array.shape[0]):
            q_left, q_ctr, q_right = triplet_array[j,:]
            adj=self.adj_dict[(q_ctr, q_left, q_right)]
            next_dl_out = [conformalize_step_dist(next_dl_out[i], q_ctr, q_left,q_right, adj) 
                       for i in range(len(next_dl_out))]
        return(next_dl_out)


def conformalize_distribution_list(dl_to_conform, dl_calib, y_calib, depth):
    triplet_array = make_triplet_array(depth)
    next_dl_calib = [dl_calib[i] for i in range(len(y_calib))]
    next_dl_to_conform = [dl_to_conform[i] for i in range(len(dl_to_conform))]
    for j in range(triplet_array.shape[0]):
        q_left, q_ctr, q_right = triplet_array[j,:]
        print(f'calibrating {q_ctr} with boundaries {q_left, q_right}')
        resids = y_calib - np.array([next_dl_calib[i].quantile(q_ctr) for i in range(len(y_calib))])
        adj = np.quantile(resids, q_ctr)
        print(f'adjustment is {adj}')
        next_dl_calib = [conformalize_step_dist(next_dl_calib[i], q_ctr, q_left,q_right, adj) 
                   for i in range(len(y_calib))]
        next_dl_to_conform = [conformalize_step_dist(next_dl_to_conform[i], q_ctr, q_left,q_right, adj) 
                   for i in range(len(next_dl_to_conform))]
    return(next_dl_to_conform)


def conformalize_distribution_list_slide_stretch(dl_to_conform, dl_calib, y_calib, slide_q, 
                                                 left_qs, right_qs):
    next_dl_calib = [dl_calib[i] for i in range(len(y_calib))]
    resids_init = y_calib - np.array([next_dl_calib[i].quantile(slide_q) for i in range(len(y_calib))])
    adj_init = np.quantile(resids_init, slide_q)
    print(f'initial adjustment is {adj_init}, sliding distributions')
    next_dl_calib = [conformalize_slide_dist(next_dl_calib[i], adj_init) 
                   for i in range(len(y_calib))]
    next_dl_to_conform = [conformalize_slide_dist(dl_to_conform[i] , adj_init) for i in range(len(dl_to_conform))]
    left_qs = -np.sort(-np.array(left_qs))
    right_qs = np.sort(right_qs)
    last_q = slide_q
    for q in left_qs:
        print(f'calibrating {q} fixing {last_q}')
        resids = y_calib - np.array([next_dl_calib[i].quantile(q) for i in range(len(y_calib))])
        adj = np.quantile(resids, q)
        print(f'adjustment is {adj}')
        next_dl_calib = [conformalize_stretch_dist(next_dl_calib[i], q, last_q, adj) 
                   for i in range(len(y_calib))]
        next_dl_to_conform = [conformalize_stretch_dist(next_dl_to_conform[i], q, last_q, adj) 
                   for i in range(len(next_dl_to_conform))]
        last_q=q
    last_q = slide_q
    for q in right_qs:
        print(f'calibrating {q} fixing {last_q}')
        resids = y_calib - np.array([next_dl_calib[i].quantile(q) for i in range(len(y_calib))])
        adj = np.quantile(resids, q)
        print(f'adjustment is {adj}')
        next_dl_calib = [conformalize_stretch_dist(next_dl_calib[i], q, last_q, adj) 
                   for i in range(len(y_calib))]
        next_dl_to_conform = [conformalize_stretch_dist(next_dl_to_conform[i], q, last_q, adj) 
                   for i in range(len(next_dl_to_conform))]
        last_q=q
    return(next_dl_to_conform)


def conformalize_slide_dist(dist, adj):
    # adj should be the q_ctr quantile of the residuals between
    # the true answers on the calibration set and the q_ctr quantiles
    # of the predicted distributions
    new_binvec = dist.binvec.copy()+adj
    new_probvec = dist.probvec.copy()
    return(PdfDiscrete(new_binvec, new_probvec))

def conformalize_stretch_dist(dist, q_adj, q_fixed, adj):
    # in this variant, we want to move the quantile from q_adj 
    # to q_adj+adj, but keep q_fixed in the same place.
    # so if q_adj<q_fixed, we stretch to the left
    # otherwise we stretch to the right
    if (q_adj<q_fixed):
        if adj>(q_fixed-q_adj):
            print("adjustment reorders quantiles, leaving unadjusted")
            return(dist)
    if (q_adj>q_fixed):
        if adj<(q_fixed-q_adj):
            print("adjustment reorders quantiles, leaving unadjusted")
            return(dist)
    curr_q_adj_quantile = dist.quantile(q_adj)
    curr_q_fixed_quantile = dist.quantile(q_fixed)
    mod_dist = dist.add_binpts([curr_q_fixed_quantile])
    new_fixed_bpi = np.where(mod_dist.binvec == curr_q_fixed_quantile)[0][0]
    new_binvec = mod_dist.binvec.copy()
    new_probvec = mod_dist.probvec.copy()
    
    if (q_adj<q_fixed):
        ratio = ((curr_q_fixed_quantile-curr_q_adj_quantile)-adj)/(curr_q_fixed_quantile-curr_q_adj_quantile)
        inters = curr_q_fixed_quantile-new_binvec[:new_fixed_bpi]
        new_inters = inters*ratio
        new_binvec[:new_fixed_bpi] = curr_q_fixed_quantile -new_inters
    if (q_adj>q_fixed):
        ratio = ((curr_q_fixed_quantile-curr_q_adj_quantile)-adj)/(curr_q_fixed_quantile-curr_q_adj_quantile)
        inters = new_binvec[new_fixed_bpi+1:]-curr_q_fixed_quantile
        new_inters = inters*ratio
        new_binvec[new_fixed_bpi+1:] = curr_q_fixed_quantile+new_inters
    return(PdfDiscrete(binvec=new_binvec, probvec=new_probvec))


def conformalize_step_dist(dist, q_ctr, q_left, q_right, adj):
    # adj should be the q_ctr quantile of the residuals between
    # the true answers on the calibration set and the q_ctr quantiles
    # of the predicted distributions
    curr_q_ctr_quantile = dist.quantile(q_ctr)
    curr_q_left_quantile = dist.quantile(q_left)
    curr_q_right_quantile = dist.quantile(q_right)
    ctr_binpt_insert = dist.quantile(q_ctr) + adj
    left_binpt_insert = curr_q_left_quantile
    right_binpt_insert = curr_q_right_quantile
    if((ctr_binpt_insert<curr_q_left_quantile) or (ctr_binpt_insert>curr_q_right_quantile)):
        print("adjustment crosses boundary, leaving unchanged")
        return(dist)
    else:
        mod_dist = dist.add_binpts([left_binpt_insert, ctr_binpt_insert, right_binpt_insert])
        left_interval_prob = mod_dist.cdf([ctr_binpt_insert])[0] - mod_dist.cdf([left_binpt_insert])[0]
        mult_adj_left = (q_ctr-q_left)/left_interval_prob
        mult_adj_right = (q_ctr-q_left)/(q_right-q_left - left_interval_prob)
        new_ctr_bpi = np.where(mod_dist.binvec == ctr_binpt_insert)[0][0]
        new_left_bpi = np.where(mod_dist.binvec == left_binpt_insert)[0][0]
        new_right_bpi = np.where(mod_dist.binvec == right_binpt_insert)[0][0]
        new_probvec = mod_dist.probvec.copy()
        new_probvec[new_left_bpi:new_ctr_bpi] = mod_dist.probvec[new_left_bpi:new_ctr_bpi] * mult_adj_left
        new_probvec[new_ctr_bpi:new_right_bpi] = mod_dist.probvec[new_ctr_bpi:new_right_bpi] * mult_adj_right
        if (np.abs(1-np.sum(new_probvec))>.0001):    
            print(f"curr_quantiles = {(curr_q_left_quantile,curr_q_ctr_quantile,curr_q_right_quantile)}")
            print(f"ctr_pt to insert = {ctr_binpt_insert}")
            print(f"binbpt_indices={new_left_bpi,new_ctr_bpi, new_right_bpi}")
            print(left_interval_prob, mult_adj_left, mult_adj_right)
            print(np.sum(mod_dist.probvec[new_left_bpi:new_ctr_bpi]), 
                              np.sum(mod_dist.probvec[new_ctr_bpi:new_right_bpi]))
        return(PdfDiscrete(mod_dist.binvec, new_probvec))

def get_adjustment(preds, y_true, qval):
    resids = y_true-preds.quantile(qval)
    return(np.quantile(resids, qval))


def make_triplet_array(depth):
    triplet_array = np.zeros((2**(depth)-1,3))
    k=0
    for i in range(1,depth+1):
        start_pt = .5**i
        triplet = np.array([0, start_pt, 2*start_pt])
        triplet_array[k,:] = triplet
        k+=1
        for j in range(2**(i-1)-1):     
            triplet = triplet+(2)*start_pt
            triplet_array[k,:] = triplet
            k+=1
    return(triplet_array)


def pdf_group_from_pdf_list(pdflist):
    full_binpts = pdflist[0].binvec
    print(len(full_binpts))
    for i in range(1,len(pdflist)):
        print(len(full_binpts))
        full_binpts = np.unique(np.concatenate((full_binpts, pdflist[i].binvec)))
    return(full_binpts)
