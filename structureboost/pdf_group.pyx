# cython: profile=True
# cython: language_level=3

import numpy as np
from pdf_discrete import PdfDiscrete
import warnings

cimport numpy as np
np.import_array()
cimport cython

class PdfGroup(object):
    """Represents a group of PdfDiscrete objects with the same bin points.

    This is the output of the `predict_distributions` method for the
    associated probabilistic regression models.

    The underlying PdfDiscrete objects can be referenced by index, e.g.

    >> pdfg = PdfGroup(...)
    >> pdf_5 = pdfg[5]
    >> pdf_5.plot_density() ## or whatever you want to do

    Parameters
    ----------

    binvec : np.array-like
        The bin points which represent the discrete intervals over which the
        probability density is defined.  Should be given in ascending order.
        A binvec with n points implies n-1 distinct intervals with the given
        probabilities / (probability densities).

    probvec : np.array-like
        The corresponding probabilities of each interval, in a matrix where the rows
        represent the individual distribtions.  Can be set to None
        if densities are provided instead.  Only one of probmat and densitymat
        should be provided.

    densitymat : np.array-like
        The corresponding densities of each interval, in a matrix where the rows
        represent the individual distribtions.  Will be ignored if the
        probmat is provided, but is necessary if not provided.

    ptol : float, default is 1e-6
        The tolerance used when checking if the probabilities sum to 1.

    check_sum : bool, default is True
        Whether or not to check that the probabilities sum to 1.  Default is
        True.  Can be turned off for computational efficiency.
"""
    def __init__(self, binvec, probmat=None, densitymat=None,
                ptol=1e-6, check_sum=True):
        self.binvec = np.array(binvec)
        self.num_bins = len(binvec)-1
        if probmat is not None:
            self.probmat = probmat
            if probmat.shape[1]!=self.num_bins:
                warnings.warn('Number of probabilities does not match number of bins.')
            if check_sum:
                if not np.allclose(np.abs(np.sum(self.probmat, axis=1)),1.0,ptol,ptol):
                    warnings.warn('The Probabilities do not sum to 1.')
            self.bin_widths = self.binvec[1:] - self.binvec[:-1]
            self.densitymat = self.probmat / self.bin_widths
            if densitymat is not None:
                warnings.warn('Both probs and densities given, ignoring densities.')
        elif densitymat is not None:
            if densitymat.shape[1]!=self.num_bins:
                warnings.warn('Number of columns of densitymat does not match number of bins.')
            self.densitymat = densitymat
            self.bin_widths = self.binvec[1:] - self.binvec[:-1]
            self.probmat = self.densitymat * self.bin_widths
            if check_sum:
                if not np.allclose(np.abs(np.sum(self.probmat, axis=1)),1.0,ptol,ptol):
                    warnings.warn('The Probabilities do not sum to 1.')
        self.bin_midpts = (self.binvec[:-1]+self.binvec[1:])/2
        self.support_min, self.support_max = self.binvec[0], self.binvec[-1]
        self.pdf_list = None
        self.cum_prob_mat = None

    def calculate_cum_prob_mat(self):
        nrows = self.probmat.shape[0]
        self.cum_prob_mat = np.cumsum(np.hstack((np.zeros(nrows).reshape(-1,1), 
                                    self.probmat)), axis=1)

    def make_pdf_list(self, check=False):
        self.pdf_list = [PdfDiscrete(self.binvec, self.probmat[i,:], check_sum=False)
                         for i in range(self.probmat.shape[0])]

    def __getitem__(self, arg):
        if self.pdf_list is None:
            return(PdfDiscrete(self.binvec, self.probmat[arg,:]))
        else:
            return(self.pdf_list[arg])

    def log_loss(self, y_test, eps=1e-16):
        """Computes the log-loss (Negative log likelihood) based on prob density."""
        if type(y_test)!= np.ndarray:
            y_test = np.array(y_test)
        if self.densitymat.shape[0]!=len(y_test):
            warnings.warn("Length of y_test != number of pdfs")
        if self.pdf_list is None:
            self.make_pdf_list()
        density_vals = np.array(
            [pdf.density([y_test[i]]) for i,pdf in enumerate(self.pdf_list)])
        density_vals = np.maximum(density_vals, eps)
        return(-np.mean(np.log(density_vals)))

    def crps_mean(self, y_test, eps=1e-16):
        """Computes the log-loss (Negative log likelihood) based on prob density."""
        if type(y_test)!= np.ndarray:
            y_test = np.array(y_test)
        if self.densitymat.shape[0]!=len(y_test):
            warnings.warn("Length of y_test != number of pdfs")
        if self.pdf_list is None:
            self.make_pdf_list()
        crps_vals = np.array(
            [pdf.crps_single_pt(y_test[i]) for i,pdf in enumerate(self.pdf_list)])
        return(np.mean(crps_vals))

    def mean(self):
        "Returns an array of means of the associated pdfs"
        if self.pdf_list is None:
            self.make_pdf_list()
        return(np.array([pdf.mean() for pdf in self.pdf_list]))

    def median(self):
        "Returns an array of medians of the associated pdfs"
        if self.pdf_list is None:
            self.make_pdf_list()
        return(np.array([pdf.median() for pdf in self.pdf_list]))

    def quantile(self, val):
        "Returns the `val`th quantile of the associated pdfs"
        if self.pdf_list is None:
            self.make_pdf_list()
        return(np.array([pdf.quantile(val) for pdf in self.pdf_list]))

    def quantiles(self, vals):
        "Returns a matrix with the specified quantiles of the associated pdfs"
        if self.pdf_list is None:
            self.make_pdf_list()
        return(np.array([pdf.quantiles(vals) for pdf in self.pdf_list]))

    def test_between_quantiles(self, test_vals, q_left, q_right):
        "Returns boolean array indicating inclusion of test values between quantiles"
        if self.pdf_list is None:
            self.make_pdf_list()
        return((test_vals>=self.quantile(q_left)) & (test_vals<=self.quantile(q_right)))

    def test_in_pred_regions(self, test_vals, alpha):
        if self.pdf_list is None:
            self.make_pdf_list()
        return(test_in_pred_regions(test_vals, self.pred_regions(alpha)))

    def quantile_widths(self, q_left, q_right):
        if self.pdf_list is None:
            self.make_pdf_list()
        return(self.quantile(q_right)-self.quantile(q_left))


    def pred_region_sizes(self, coverage):
        if self.pdf_list is None:
            self.make_pdf_list()
        prs = self.pred_regions(coverage)
        return(np.array([size_of_pred_region(pr) for pr in prs]))


    def pred_regions(self, coverage):
        if self.pdf_list is None:
            self.make_pdf_list()
        return(np.array([pdf.pred_region(coverage) for pdf in self.pdf_list], dtype=object))

    def bins_to_probs(self, new_binpt_vec, eps=1e-16):
        """Assumes new_binpt_vec contains same first and last point"""
        no_new_binpts = np.all(np.isin(new_binpt_vec, self.binvec))
        numrows = self.densitymat.shape[0]
        if not no_new_binpts:
            finebins = np.union1d(self.binvec, new_binpt_vec)
            bigmat = explode_densities(self.densitymat, self.binvec, finebins)
            fbw = np.diff(finebins)
            bigmat = bigmat*fbw
            bigmat = np.cumsum(bigmat,axis=1)
        else:
            finebins=self.binvec
            bigmat = np.cumsum(self.probmat, axis=1)
        indices = (np.argwhere(np.isin(finebins, new_binpt_vec)).reshape(-1)-1).tolist()
        new_probs = np.diff(bigmat[:,indices[1:]], axis=1, prepend=0)
        new_probs = np.maximum(new_probs, eps)
        new_probs = new_probs/np.sum(new_probs, axis=1).reshape(-1,1)
        return(new_probs)


def test_in_pred_regions(test_vals, regions):
    return(np.array([np.any([((test_vals[i]>=interval[0]) and (test_vals[i]<=interval[1]))
                     for interval in regions[i]]) 
                        for i in range(len(test_vals))]))


def log_loss_pdf(y_true, pdf_set):
    """This computes the log loss for a truth set and a DensitySet object"""
    return pdf_set.log_loss(y_true)

def crps_mean(y_true, pdf_set):
    """This computes the log loss for a truth set and a DensitySet object"""
    return pdf_set.crps_mean(y_true)

def size_of_pred_region(pr):
    return(np.sum([x[1]-x[0] for x in pr]))


def test_between_quantiles_pdf(test_vals, pdf_set, q_left, q_right):
    """Tests inclusion of the test_vals in the interval defined by the given quantiles"""
    return(pdf_set.test_between_quantiles(test_vals, q_left, q_right))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def explode_densities(np.ndarray[double, ndim=2] densitymat, 
                               np.ndarray[double] density_binpts, 
                               np.ndarray[double] fine_binpts):

    cdef long num_rows = densitymat.shape[0]
    cdef long num_fine_bins = len(fine_binpts)-1
    cdef np.ndarray[double, ndim=2] fine_densitymat=np.zeros((num_rows, num_fine_bins))
    cdef long dens_ptr =0
    cdef long row_ptr, fine_ptr
    
    for fine_ptr in range(num_fine_bins):
        for row_ptr in range(num_rows):            
            fine_densitymat[row_ptr, fine_ptr] = densitymat[row_ptr, dens_ptr]
        if fine_binpts[fine_ptr+1] >= density_binpts[dens_ptr+1]:
            dens_ptr+=1
    return fine_densitymat

