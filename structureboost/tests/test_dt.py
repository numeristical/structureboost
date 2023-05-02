import structure_dt as stdt
import numpy as np


def test_get_mask_int_c():
    len_to_use = 10
    asd = np.arange(len_to_use).astype(np.int32)
    inq_set = np.array([3, 5, 6]).astype(np.int32)
    mv = np.zeros(len_to_use).astype(np.int32)
    response = stdt.get_mask_int_c(asd, inq_set, len(asd),
                                       len(inq_set), mv)
    answer = np.array([False, False, False, True, False, True, True,
                       False, False, False])
    assert np.sum(answer == response) == len_to_use

def test_get_bin_sums_c():
    asd = np.linspace(.0001,.9999,9999)
    zxc = np.linspace(0,1,11)
    my_g = .1*np.ones(9999)
    my_h = .2*np.ones(9999)
    my_g_h = np.vstack((my_g, my_h)).T
    brv = np.searchsorted(zxc,asd).astype(np.int32)
    answer_g, answer_h = stdt.get_bin_sums_c(my_g_h, brv,12)
    assert (np.round(answer_g[9],1)==100) and (np.round(answer_h[10],1)==199.8)
