import numpy as np
import pandas as pd
import structureboost as stb

def test_pdfgroup_init1():
    pm = np.array([[0.3, 0.5, 0.2],[0.12,0.18, 0.7]])
    bv = np.array([0.0, 3.0, 5.0, 10.0])
    pdfg = stb.PdfGroup(bv, pm)
    right_answer = np.array([[0.1, 0.25, 0.04],
    [0.04, 0.09, 0.14]])
    assert(np.allclose(pdfg.densitymat, right_answer))

def test_pdfgroup_init2():
    bv = np.array([0.0, 3.0, 5.0, 10.0])
    dm = np.array([[0.1, 0.25, 0.04],
    [0.04, 0.09, 0.14]])
    print(dm.shape)
    pdfg = stb.PdfGroup(bv, densitymat=dm)
    right_answer = np.array([[0.3, 0.5, 0.2],[0.12,0.18, 0.7]])
    assert(np.allclose(pdfg.probmat, right_answer))
