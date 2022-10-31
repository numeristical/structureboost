# cython: profile=True
# cython: language_level=3

import warnings
import numpy as np
import matplotlib.pyplot as plt

class PdfDiscrete(object):
    
    def __init__(self, binvec, probvec, ptol=1e-6):
        self.binvec = np.array(binvec)
        self.probvec = np.array(probvec)
        self.num_bins = len(binvec)-1
        if len(self.probvec)!=self.num_bins:
            warnings.warn('Number of probabilities does not match number of bins.')
        if np.abs(np.sum(self.probvec)-1.0)>ptol:
            warnings.warn('Probabilities do not sum to 1.')
        self.bin_midpts = (self.binvec[:-1]+self.binvec[1:])/2
        self.bin_widths = self.binvec[1:] - self.binvec[:-1]
        self.support_min, self.support_max = self.binvec[0], self.binvec[-1]
        self.bin_densities = self.probvec / self.bin_widths
        self.cum_prob = np.concatenate(([0],np.cumsum(self.probvec)))
    
    def mean(self):
        val = np.sum(self.bin_midpts*self.probvec)
        return(val)

    def quantile(self, qval):
        bin_lp = np.digitize(qval, self.cum_prob)-1
        
        if bin_lp>=len(self.binvec)-1:
            bin_lp = len(self.binvec)-2
        binpt_left = self.binvec[bin_lp]
        cumprob_left = self.cum_prob[bin_lp]
        out_val =  binpt_left + self.bin_widths[bin_lp]*(qval-cumprob_left)/self.probvec[bin_lp]
        return(out_val)

    def quantiles(self, qval):
        bin_lp = np.digitize(qval, self.cum_prob)-1
        bin_lp[bin_lp>=len(self.binvec)-1] = len(self.binvec)-2
        binpt_left = self.binvec[bin_lp]
        cumprob_left = self.cum_prob[bin_lp]
        out_val =  binpt_left + self.bin_widths[bin_lp]*(qval-cumprob_left)/self.probvec[bin_lp]
        return(out_val)
    
    def median(self):
        return(self.quantile(.5))
        
    def density(self, value_vec, include_right=True):
        bin_ind = np.digitize(value_vec, self.binvec)
        if include_right:
            bin_ind[value_vec==self.support_max] = self.num_bins
        aug_dens_vec = np.concatenate(([0],self.bin_densities,[0]))
        return(np.array([aug_dens_vec[bin_ind[i]] for i in range(len(bin_ind))]))

    def cdf(self, x_vals):
        bin_ind = np.digitize(x_vals, self.binvec)
        aug_bin_vec = np.concatenate(([-np.inf],self.binvec))
        aug_cum_prob = np.concatenate(([0],self.cum_prob))
        aug_densities = np.concatenate(([0],self.bin_densities,[0]))
        remainder_x = x_vals - aug_bin_vec[bin_ind]
        remainder_x[np.isinf(remainder_x)] = 0
        out_vec = aug_cum_prob[bin_ind] +  remainder_x * aug_densities[bin_ind]
        return(out_vec)

    def __add__(self, pd2):
        new_binvec = np.unique(np.concatenate((self.binvec, pd2.binvec)))
        new_midpts = (new_binvec[1:] + new_binvec[:-1])/2
        pv1 = self.density(new_midpts)*self.bin_widths
        pv2 = pd2.density(new_midpts)*pd2.bin_widths
        return(PdfDiscrete(new_binvec, (pv1+pv2)/2))

    
    def plot_density(self, pred_alpha=None, 
                    pred_type='interval', pred_color='orange',
                    **kwargs):
        density_plot(self.binvec, self.probvec, **kwargs)
        if pred_alpha is not None:
            if pred_type=='region':
                included_intervals = self.pred_region_bins(pred_alpha)
                for index in included_intervals:
                    plt.bar(x=self.bin_midpts[index], height=self.bin_densities[index],
                        width = self.bin_widths[index], color=pred_color)
            if pred_type=='interval':
                left_pi, right_pi = np.digitize([.05, .95], self.cum_prob)-1
                left_xpt, right_xpt = self.quantiles([.05,.95])
                included_intervals = list(range(left_pi+1, right_pi))
                left_midpt = (left_xpt + self.binvec[left_pi+1])/2
                left_width = self.binvec[left_pi+1] -left_xpt
                right_midpt = (right_xpt + self.binvec[right_pi])/2
                right_width = right_xpt - self.binvec[right_pi]
                self.plot_density()
                for index in included_intervals:
                    plt.bar(x=self.bin_midpts[index], height=self.bin_densities[index],
                        width = self.bin_widths[index], color=pred_color)

                plt.bar(x=left_midpt, height=self.bin_densities[left_pi],
                        width=left_width, color=pred_color)
                plt.bar(x=right_midpt, height=self.bin_densities[right_pi],
                        width=right_width, color=pred_color)


    def plot_cdf(self, **kwargs):
        plt.plot(self.binvec, self.cum_prob, **kwargs)

    def pred_region(self, alpha):
        order = np.argsort(-self.bin_densities)
        cum_prob = np.cumsum(self.probvec[order])
        index = np.min(np.where(cum_prob>=alpha))
        inc_int = np.sort(order[:index+1])

        num_ints = len(inc_int)
        jkl = 0
        int_list = []
        while (jkl<(num_ints-1)):
            interval_index = inc_int[jkl]
            left_pt = self.binvec[interval_index]

            # advance until the intervals are non-consecutive
            while (jkl<(num_ints-1)) and (inc_int[jkl+1] == (inc_int[jkl]+1)):
                jkl+=1
            right_pt = self.binvec[inc_int[jkl]+1]
            int_list.append((left_pt, right_pt))
            jkl+=1
        return(int_list)

    def pred_region_bins(self, alpha):
        order = np.argsort(-self.bin_densities)
        cum_prob = np.cumsum(self.probvec[order])
        index = np.min(np.where(cum_prob>=alpha))
        inc_int = np.sort(order[:index+1])
        return(inc_int)

    def pred_interval(self, alpha):
        return((self.quantile(alpha/2), self.quantile(1-(alpha/2))))

    # def plot_density_with_pr(self, alpha, bar_color='black', **kwargs):
    #     self.plot_density(**kwargs)
    #     included_intervals = self.confidence_region_bins(alpha)
    #     for index in included_intervals:
    #         plt.bar(x=self.bin_midpts[index], height=self.bin_densities[index],
    #             width = self.bin_widths[index], color=bar_color)

def get_part(nc, blocksize, start_val):
    list1=[]
    num_full_blocks = int((nc-start_val)/blocksize)
    upper_lim = num_full_blocks*blocksize+start_val
    if start_val>0:
        list1.append(list(range(0,start_val)))
    list1+=(list(range(i*blocksize+start_val, (i+1)*blocksize+start_val)) for i in range(num_full_blocks))
    if upper_lim<nc:
        list1.append(list(range(upper_lim, nc)))
    return(list1)

def chain_partition(nc, blocksize):
    part = [get_part(nc, blocksize, sv) for sv in range(blocksize)]
    return(part)

def density_plot(bin_vec, prob_vec, **kwargs):
    binvec_plot = np.vstack((bin_vec, bin_vec)).T.reshape(-1)
    prob_dens = prob_vec / np.diff(bin_vec)
    prob_plot = np.concatenate(([0],np.vstack((prob_dens, prob_dens)).T.reshape(-1),[0]))
    plt.plot(binvec_plot, prob_plot, **kwargs)

def average_densities(pd_list):
    binpt_list = [pd.binvec for pd in pd_list]
    new_binvec = np.unique(np.concatenate(binpt_list))
    new_bin_widths = new_binvec[1:] - new_binvec[:-1]
    new_midpts = (new_binvec[1:]+new_binvec[:-1])/2
    num_dens = len(pd_list)
    num_bins = len(new_midpts)
    prob_array = np.zeros((num_dens, num_bins))
    for i,pd in enumerate(pd_list):
        prob_array[i,:] = pd.density(new_midpts)*new_bin_widths
    new_probvec = np.mean(prob_array, axis=0)
    # return(prob_array)
    return(PdfDiscrete(new_binvec, new_probvec))    
