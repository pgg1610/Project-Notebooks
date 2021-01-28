PyBO: Bayesian Optimization in Python

This package provides simple routines that leverage the capabilities of the
sklearn.gaussian_process.GaussianProcessRegressor object in order to perform
Bayesian optimization of scalar functions. PyBO is not directly dependent on
the sklearn package, in the sense that the command ``import pybo'' will not
attempt to access sklearn in any way. Instead, the function definitions within
PyBO assume that certain objects have the same methods and attributes as the
GaussianProcessRegressor.



Dependencies
------------
	numpy
	scipy
	tqdm
	warnings



Installation
------------
At the moment, there is no proper installation executable. Instead, copy the
``pybo'' directory (see folder hierarchy, below) either to a location on your
PYTHONPATH or to the same directory as any code that needs to import it.



Folder Hierarchy
----------------
+
|----+ pybo
|    |---- __init__.py
|    |---- acquisition.py
|    |---- objectives.py
|    |---- opti.py
|    \---- utils.py
|---- demo.ipynb
|---- demo.py
|---- LICENSE.txt
\---- README.txt



License
-------
This code is released under the MIT license. See LICENSE.txt for details.