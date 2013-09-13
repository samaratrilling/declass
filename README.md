DDRS
====

Style Guide
-----------

Read the [LowClass Python style-guide](http://columbia-applied-data-science.github.io/pages/lowclass-python-style-guide.html)

Dependencies
------------
`utils` repo 

The directory above this repo (as well as `utils`) should be added to your `PYTHONPATH`.

Data
----
* Choose your own data location
* Don't commit data to the repo

Read README_data.md for more

Directories
-----------

### notebooks
For ipython notebooks.  Put your name in the notebook name to avoid redundancy.

### src
Source code.

### schema
Schema for the data sets

### tests
Unit and integration tests.

### scripts
Shell scripts, python scripts, etc...


Server
======

To add packages using Chef
--------------------------

    open up /opt/chef/jrl/site-cookbooks/XXX/recipes/default.rb
    with any editor
    and add the packages you want
    the script runs on a 10min chron
    you can also just run it
    sudo chef-solo -c /opt/chef/jrl/solo.rb

