"""Setup for the structureboost package."""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[Extension("graphs",
                                      ["structureboost/graphs.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_dt",
                                      ["structureboost/structure_dt.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_gb",
                                      ["structureboost/structure_gb.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_gb_multi",
                                      ["structureboost/structure_gb_multi.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("pdf_discrete",
                                      ["structureboost/pdf_discrete.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("pdf_group",
                                      ["structureboost/pdf_group.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("prob_regr_unit",
                                      ["structureboost/prob_regr_unit.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("prob_regressor",
                                      ["structureboost/prob_regressor.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("coarsage",
                                      ["structureboost/coarsage.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_dt_multi",
                                      ["structureboost/structure_dt_multi.c"],
                                      include_dirs=[numpy.get_include()])],
)
