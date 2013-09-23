declass
=======
Code for the Declassification project.

Style Guide
-----------

* Make changes through pull requests
* Submit issues and bug reports with GitHub issues
* Start reading the [LowClass Python style-guide](http://columbia-applied-data-science.github.io/pages/lowclass-python-style-guide.html)

Performance & Style
-------------------
* Good programming style can lead to faster as well as cleaner code
* Refer to the python [performance tips page](https://wiki.python.org/moin/PythonSpeed/PerformanceTips) for more info and general useful reading 

Dependencies
------------

* [parallel_easy](https://github.com/langmore/parallel_easy.git)
* A number of standard python modules that can be installed using `pip`
* The easiest way of installing/maintaining many python packages is by using [anaconda](https://store.continuum.io/cshop/anaconda/).

This directory (and all depencencies) should be added to your `PYTHONPATH`.

Data
----
* Choose your own data location
* Don't commit data to the repo

Read `README_data.md` for more.

Directories
-----------

### notebooks
For ipython notebooks.  Put your name in the notebook name to avoid redundancy.

### schema
Schema for the data sets

### scripts
Shell scripts, python scripts, etc...


Code Directories
----------------

### declass
modules specific to declassification project

### declass/utils
general utility modules

### declass/cmd
command-line utilities

### declass/tests
Unit tests for declass.
