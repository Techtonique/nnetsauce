
Installation
''''''''''

`nnetsauce` is currently available for Python and R. 

For Python
----------


**Stable version** From Pypi: 

.. code-block:: console

    pip install nnetsauce


**Development version** From GitHub. For this to work, you'll first need to have `Git installed <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ : 

.. code-block:: console

    pip install git+https://github.com/thierrymoudiki/nnetsauce.git


For R
-----

In R console, type: 

.. code-block:: r

	install.packages("devtools") # if devtools isn't installed yet
	library(devtools)
	devtools::install_github("thierrymoudiki/nnetsauce/R-package")
	library(nnetsauce)

**General rule for using the package in R** (**the API is identical**):  object accesses with `.`'s are replaced by `$`'s. For **R examples**, you can type the following command in R console:

.. code-block:: r

	# CustomRegressor is a nnetsauce class
	help("CustomRegressor")


And read section **Examples** in the Help page displayed. Any other class name of the API Documentation (see sections :ref:`ref-regression-models`, :ref:`ref-classification-models` and :ref:`ref-time-series-models`) can be used instead of ``CustomRegressor``. 

Next section presents some **examples of use of nnetsauce** in Python, that are representative of its general philosophy, but not exhaustive. For more examples, you can refer to `these notebooks <https://github.com/thierrymoudiki/nnetsauce/tree/master/nnetsauce/demo>`_ or `this directory <https://github.com/thierrymoudiki/nnetsauce/tree/master/examples>`_; feel free to contribute yours in there too, with the following naming convention:  ``yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]``.