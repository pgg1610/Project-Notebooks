# -*- coding: utf-8 -*-
"""
pybo: Bayesian Optimization in Python

This package provides simple routines that leverage the capabilities of the
sklearn.gaussian_process.GaussianProcessRegressor object in order to perform
Bayesian optimization of scalar functions.

"""
from . import acquisition
from . import objectives

from .opti import bayesian_optimization