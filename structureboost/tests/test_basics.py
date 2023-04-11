import numpy as np
import structureboost as stb
import coarsage
import pdf_group as pdfg

def test_update_densities_logprobs_1():
    densitymat = np.array([[1,1,1,1,1,1],[1.5,.5,1.5,.5,1.5,.5]])/6
    density_binpts = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0])
    deltamat = np.array([[1.0,-1.0,1.0,-1.0],[-1.0,1.0,-1.0,1.0]])
    delta_binpts = np.array([0.0,2.5,3.5,5.0,6.0])
    fine_binpts = np.unique(np.concatenate((density_binpts, delta_binpts))) 
    fine_bin_widths = np.diff(fine_binpts)

    outmat = coarsage.update_densities_to_logprobs(densitymat, 
                               density_binpts, 
                               deltamat, 
                               delta_binpts,
                               fine_binpts,
                               fine_bin_widths,
                               0.1)

    right_answer = np.array([[-1.69175947, -1.69175947, -2.38490665, -2.58490665, -2.58490665, 
        -2.38490665,-1.69175947, -1.89175947],
     [-1.48629436, -2.58490665, -2.17944154, -1.97944154, -3.07805383, -3.27805383,
      -1.48629436, -2.38490665,]])
    print(outmat)
    assert(np.allclose(outmat, right_answer))

def test_tensor_result_sum_fine_1():
    coarse_pred = np.arange(2*3*4).reshape(2,3,4)*1.0
    bv_mat = np.array([[0.0,2.0,3.0,5.0,6.0],
                       [0.0,3.0,4.0,5.0,6.0],
                       [0.0,1.0,2.0,3.0,6.0]])
    fine_binpts = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0])
    lr=0.1

    outmat = coarsage.tensor_result_sum_fine(coarse_pred,
                           bv_mat,fine_binpts, lr)
    right_answer = np.array([[1.2, 1.3, 1.5, 1.8, 1.9, 2.1],
                             [4.8, 4.9, 5.1, 5.4, 5.5, 5.7]])
    assert(np.allclose(outmat, right_answer))

def test_explode_densities_1():
    dm = np.array([[1.0,2.0,3.0,2.0],
                   [3.0,0.5,2.0,3.0]])
    dens_binpts = np.array([0.0,1.0,3.0,4.0,5.0])
    fine_binpts = np.array([0.0,0.75,1.0,1.5,2.0,3.0,4.0,4.99,5.0])

    outmat = pdfg.explode_densities(dm, dens_binpts, fine_binpts)
    right_answer = np.array([[1.,1.,2.,2.,2.,3.,2,2.],
                             [3.,3.,0.5,0.5,0.5,2.,3.,3.]])
    assert(np.allclose(outmat, right_answer))



