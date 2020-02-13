"""Setup for the structureboost package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Brian Lucena",
    author_email="brianlucena@gmail.com",
    name='structureboost',
    license="MIT",
    description='structureboost is a Python package for gradient boosting usin categorical structure',
    version='0.0.1',
    long_description=README,
    url='https://github.com/numeristical/structureboost',
    packages=setuptools.find_packages('structureboost/__init__.py'),
    package_dir={'structureboost':
                 'structureboost'},
    python_requires=">=3.5",
    install_requires=[
        "pandas",
        "numpy"],
    ext_modules = [setuptools.Extension("graphs", ["structureboost/graphs.c"])],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
