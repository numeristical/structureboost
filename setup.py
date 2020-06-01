"""Setup for the structureboost package."""

import setuptools
import numpy

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Brian Lucena",
    author_email="brianlucena@gmail.com",
    name='structureboost',
    license="MIT",
    description="""StructureBoost is a Python package for gradient boosting
                   using categorical structure.  See documentation at:
                   https://structureboost.readthedocs.io/""",
    version='0.0.7',
    long_description=README,
    url='https://github.com/numeristical/structureboost',
    packages=['structureboost'],
    package_dir={'structureboost':
                 'structureboost'},
    python_requires=">=3.5",
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.16",
        "scipy>=1.3"],
    ext_modules=[setuptools.Extension("graphs", ["structureboost/graphs.c"],
                 include_dirs=[numpy.get_include()]),
                 setuptools.Extension("structure_dt",
                                      ["structureboost/structure_dt.c"],
                                      include_dirs=[numpy.get_include()]),
                 setuptools.Extension("structure_gb",
                                      ["structureboost/structure_gb.c"],
                                      include_dirs=[numpy.get_include()]),
                 setuptools.Extension("structure_gb_mc",
                                      ["structureboost/structure_gb_mc.c"],
                                      include_dirs=[numpy.get_include()]),
                 setuptools.Extension("structure_gb_multiclass",
                                      ["structureboost/structure_gb_multiclass.c"],
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
    ]
)
