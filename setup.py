"""Setup for the structureboost package."""

from setuptools import Extension, setup
import numpy

with open('README.md') as f:
    README = f.read()

requirements = [
        "pandas>=1.0",
        "numpy>=1.16",
        "scipy>=1.3",
        "matplotlib>=1.0",
        "joblib>=0.14.1",
        "ml_insights>=1.0.0"]

setup(
    author="Brian Lucena",
    author_email="brianlucena@gmail.com",
    name='structureboost',
    license="MIT",
    license_files=['LICENSE'],
    description="""StructureBoost is a Python package for gradient boosting
                   using categorical structure.  See documentation at:
                   https://structureboost.readthedocs.io/""",
    version='0.3.1',
    long_description=README,
    zip_safe=False,
    url='https://github.com/numeristical/structureboost',
    packages=['structureboost'],
    package_dir={'structureboost':
                 'structureboost'},
    python_requires=">=3.5",
    install_requires=requirements,
    ext_modules=[Extension("graphs",
                                      ["structureboost/graphs.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_dt",
                                      ["structureboost/structure_dt.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_rfdt",
                                      ["structureboost/structure_rfdt.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_rf",
                                      ["structureboost/structure_rf.c"],
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
                 Extension("pdf_set",
                                      ["structureboost/pdf_set.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("prob_regr_unit",
                                      ["structureboost/prob_regr_unit.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("prob_regressor",
                                      ["structureboost/prob_regressor.c"],
                                      include_dirs=[numpy.get_include()]),
                 Extension("structure_dt_multi",
                                      ["structureboost/structure_dt_multi.c"],
                                      include_dirs=[numpy.get_include()])],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
