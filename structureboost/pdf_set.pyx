# cython: profile=True
# cython: language_level=3

import numpy as np

class PdfSet(object):
    def __init__(self, pdf_list):
        self.pdf_list = pdf_list

    def __getitem__(self, arg):
        return(self.pdf_list[arg])

    def len(self):
        return(len(self.pdf_list))

    def log_loss(self, y_test, eps=1e-16):
        if type(y_test)!= np.ndarray:
            y_test = np.array(y_test)
        density_vals = np.array(
            [pdf.density([y_test[i]]) for i,pdf in enumerate(self.pdf_list)])
        density_vals = np.maximum(density_vals, eps)
        return(-np.mean(np.log(density_vals)))

    def mean(self):
        return(np.array([pdf.mean() for pdf in self.pdf_list]))

    def median(self):
        return(np.array([pdf.median() for pdf in self.pdf_list]))

    def quantile(self, val):
        return(np.array([pdf.quantile(val) for pdf in self.pdf_list]))

    def quantiles(self, vals):
        return(np.array([pdf.quantiles(vals) for pdf in self.pdf_list]))

    def test_between_quantiles(self, test_vals, q_left, q_right):
        return((test_vals>=self.quantile(q_left)) & (test_vals<=self.quantile(q_right)))

    def test_in_pred_regions(self, test_vals, alpha):
        return(test_in_pred_regions(test_vals, self.pred_regions(alpha)))

    def quantile_widths(self, q_left, q_right):
        return(self.quantile(q_right)-self.quantile(q_left))

    def pred_regions(self, alpha):
        return(np.array([pdf.pred_region(alpha) for pdf in self.pdf_list], dtype=object))

    # def cdf():
    #     return()

    def bins_to_probs(self, new_binpt_vec):
        return(np.array([pdf.bins_to_probs(new_binpt_vec) for pdf in self.pdf_list]))

def test_in_pred_regions(test_vals, regions):
    return(np.array([np.any([((test_vals[i]>=interval[0]) and (test_vals[i]<=interval[1]))
                     for interval in regions[i]]) 
                        for i in range(len(test_vals))]))


def log_loss_pdf(y_true, pdf_set):
    """This computes the log loss for a truth set and a DensitySet object"""
    return pdf_set.log_loss(y_true)

def test_between_quantiles_pdf(test_vals, pdf_set, q_left, q_right):
    """Tests inclusion of the test_vals in the interval defined by the given quantiles"""
    return(pdf_set.test_between_quantiles(test_vals, q_left, q_right))

