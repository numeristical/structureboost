# cython: profile=True
# cython: language_level=3

import warnings
import numpy as np
import matplotlib.pyplot as plt

class PdfDiscrete(object):
    """Represents a piecewise constant probability density function.

    Parameters
    ----------

    binvec : np.array-like
        The bin points which represent the discrete intervals over which the
        probability density is defined.  Should be given in ascending order.
        A binvec with n points implies n-1 distinct intervals with the given
        probabilities / (probability densities).

    probvec : np.array-like
        The corresponding probabilities of each interval.  Can be set to None
        if densities are provided instead.  Only one of probvec and densityvec
        should be provided.

    densityvec : np.array-like
        The corresponding densities of each interval.  Will be ignored if the
        probvec is provided, but is necessary if not provided.

    ptol : float, default is 1e-6
        The tolerance used when checking if the probabilities sum to 1.

    check_sum : bool, default is True
        Whether or not to check that the probabilities sum to 1.  Default is
        True.  Can be turned off for computational efficiency.
"""
    def __init__(self, binvec, probvec=None, densityvec=None,
                 ptol=1e-6, check_sum=True):
        self.binvec = np.array(binvec)
        self.num_bins = len(binvec)-1
        if probvec is not None:
            self.probvec = probvec
            if len(self.probvec)!=self.num_bins:
                warnings.warn('Number of probabilities does not match number of bins.')
            if check_sum:
                if np.abs(np.sum(self.probvec)-1.0)>ptol:
                    warnings.warn('Probabilities do not sum to 1.')
            self.bin_widths = self.binvec[1:] - self.binvec[:-1]
            self.densityvec = self.probvec / self.bin_widths
            if densityvec is not None:
                warnings.warn('Both probs and densities given, ignoring densities.')
        elif densityvec is not None:
            if len(densityvec)!=self.num_bins:
                warnings.warn('Number of densities does not match number of bins.')
            self.densityvec = densityvec
            self.bin_widths = self.binvec[1:] - self.binvec[:-1]
            self.probvec = self.densityvec * self.bin_widths
            if check_sum:
                if np.abs(np.sum(self.probvec)-1.0)>ptol:
                    print(np.sum(self.probvec),self.probvec)
                    warnings.warn('Probabilities do not sum to 1.')
        self.bin_midpts = (self.binvec[:-1]+self.binvec[1:])/2
        self.support_min, self.support_max = self.binvec[0], self.binvec[-1]
        self.cum_prob = None

    def calculate_cum_prob(self):
        self.cum_prob = np.concatenate(([0],np.cumsum(self.probvec)))

    def mean(self):
        val = np.sum(self.bin_midpts*self.probvec)
        return(val)

    def quantile(self, qval):
        if self.cum_prob is None:
            self.calculate_cum_prob()
        bin_lp = np.digitize(qval, self.cum_prob)-1
        
        if bin_lp>=len(self.binvec)-1:
            bin_lp = len(self.binvec)-2
        binpt_left = self.binvec[bin_lp]
        cumprob_left = self.cum_prob[bin_lp]
        out_val =  binpt_left + self.bin_widths[bin_lp]*(qval-cumprob_left)/self.probvec[bin_lp]
        return(out_val)

    def quantiles(self, qval):
        if self.cum_prob is None:
            self.calculate_cum_prob()
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
        aug_dens_vec = np.concatenate(([0],self.densityvec,[0]))
        return(np.array([aug_dens_vec[bin_ind[i]] for i in range(len(bin_ind))]))

    def cdf(self, x_vals):
        # TODO: Check if I got the digitize right or if I need a -1
        bin_ind = np.digitize(x_vals, self.binvec)
        aug_bin_vec = np.concatenate(([-np.inf],self.binvec))
        if self.cum_prob is None:
            self.calculate_cum_prob()
        aug_cum_prob = np.concatenate(([0],self.cum_prob))
        aug_densities = np.concatenate(([0],self.densityvec,[0]))
        remainder_x = x_vals - aug_bin_vec[bin_ind]
        remainder_x[np.isinf(remainder_x)] = 0
        out_vec = aug_cum_prob[bin_ind] +  remainder_x * aug_densities[bin_ind]
        return(out_vec)

    def add_binpts(self, binpts_to_add):
        new_binvec = np.unique(np.concatenate((self.binvec, binpts_to_add)))
        new_midpts = (new_binvec[1:] + new_binvec[:-1])/2
        new_bin_widths = new_binvec[1:] - new_binvec[:-1]
        pv1 = self.density(new_midpts)*new_bin_widths
        return(PdfDiscrete(new_binvec, pv1))

    def bins_to_probs(self, new_binpt_vec):
        """Given a new set of binpts, return the probabilities in each bin"""
        return(np.diff(self.cdf(new_binpt_vec)))

    def plot_density(self, coverage=None, 
                    pred_type='interval', pred_color='orange',
                    **kwargs):
        basic_density_plot(self.binvec, self.densityvec, **kwargs)
        if coverage is not None:
            if pred_type=='region':
                included_intervals = self.pred_region_bins(coverage)
                for index in included_intervals:
                    plt.bar(x=self.bin_midpts[index], height=self.densityvec[index],
                        width = self.bin_widths[index], color=pred_color)
            if pred_type=='interval':
                # TODO: double check this
                if self.cum_prob is None:
                    self.calculate_cum_prob()
                beta = (1-coverage)/2
                left_pi, right_pi = np.digitize([beta, 1-beta], self.cum_prob)-1
                left_xpt, right_xpt = self.quantiles([beta,1-beta])
                included_intervals = list(range(left_pi+1, right_pi))
                left_midpt = (left_xpt + self.binvec[left_pi+1])/2
                left_width = self.binvec[left_pi+1] -left_xpt
                right_midpt = (right_xpt + self.binvec[right_pi])/2
                right_width = right_xpt - self.binvec[right_pi]
                #self.plot_density()
                basic_density_plot(self.binvec, self.densityvec, **kwargs)
                for index in included_intervals:
                    plt.bar(x=self.bin_midpts[index], height=self.densityvec[index],
                        width = self.bin_widths[index], color=pred_color)

                plt.bar(x=left_midpt, height=self.densityvec[left_pi],
                        width=left_width, color=pred_color)
                plt.bar(x=right_midpt, height=self.densityvec[right_pi],
                        width=right_width, color=pred_color)


    def plot_cdf(self, **kwargs):
        if self.cum_prob is None:
            self.calculate_cum_prob()
        plt.plot(self.binvec, self.cum_prob, **kwargs)

    def pred_region(self, alpha):
        order = np.argsort(-self.densityvec)
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

    def pred_region_bins(self, coverage):
        order = np.argsort(-self.densityvec)
        cum_prob = np.cumsum(self.probvec[order])
        index = np.min(np.where(cum_prob>=(coverage)))
        inc_int = np.sort(order[:index+1])
        return(inc_int)

    def pred_region_cdf(self, val):
        bin_num = np.digitize(val, self.binvec) - 1
        order = np.argsort(-self.densityvec)
        loc = np.where(order==bin_num)[0][0]
        cum_prob = np.cumsum(np.concatenate(([0],self.probvec[order])))
        cdf_val = (cum_prob[loc+1] + cum_prob[loc])/2
        return(cdf_val)


    def pred_interval(self, coverage):
        return((self.quantile((1-coverage)/2), self.quantile(1-((1-coverage)/2))))

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

def basic_density_plot(bin_vec, dens_vec, **kwargs):
    binvec_plot = np.vstack((bin_vec, bin_vec)).T.reshape(-1)
    dens_plot = np.concatenate(([0],np.vstack((dens_vec, dens_vec)).T.reshape(-1),[0]))
    plt.plot(binvec_plot, dens_plot, **kwargs)

def average_densities(pd_list, scaling='linear', weights=None):
    binpt_list = [pd.binvec for pd in pd_list]
    new_binvec = np.unique(np.concatenate(binpt_list))
    new_bin_widths = new_binvec[1:] - new_binvec[:-1]
    new_midpts = (new_binvec[1:]+new_binvec[:-1])/2
    num_dens = len(pd_list)
    num_bins = len(new_midpts)
    prob_array = np.zeros((num_dens, num_bins))
    if scaling=='linear':
        for i,pd in enumerate(pd_list):
            prob_array[i,:] = pd.density(new_midpts)*new_bin_widths
        if weights is None:
            new_probvec = np.mean(prob_array, axis=0)
        else:
            new_probvec = np.array(weights).dot(prob_array)
    if scaling=='log':
        for i,pd in enumerate(pd_list):
            prob_array[i,:] = pd.density(new_midpts)*new_bin_widths
            prob_array=np.maximum(prob_array, 1e-16)
            logprob_array = np.log(prob_array)
        if weights is None:
            new_logprobvec = np.mean(logprob_array, axis=0)
        else:
            new_logprobvec = np.array(weights).dot(logprob_array)
        new_tempprobvec = np.exp(new_logprobvec)
        new_probvec = new_tempprobvec/np.sum(new_tempprobvec)
    return(PdfDiscrete(new_binvec, new_probvec))

def get_bin_probs_from_data(binvec, value_vec, eps=1e-16):
    y_bin = np.digitize(value_vec,binvec[:-1]) -1
    binvals, counts = np.unique(y_bin, return_counts=True)
    probvec = np.zeros(len(binvec)-1)
    total = np.sum(counts)
    for i,binnum in enumerate(binvals):
        probvec[binnum] =  counts[i]/total
    probvec = np.maximum(probvec, eps)
    probvec = probvec / np.sum(probvec)
    return(probvec)

def get_pdf_from_data(binvec, value_vec):
    pv = get_bin_probs_from_data(binvec,value_vec)
    return(PdfDiscrete(binvec, pv))

