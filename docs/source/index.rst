.. _ref-homepage:

nnetsauce's documentation
=========================================

.. image:: https://img.shields.io/pypi/v/nnetsauce
      :target: https://pypi.org/project/nnetsauce/
      :alt: Latest PyPI version


.. image:: https://img.shields.io/pypi/l/nnetsauce
      :target: https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE   
      :alt: PyPI - License


.. image:: https://pepy.tech/badge/nnetsauce
      :target: https://pepy.tech/project/nnetsauce
      :alt: Number of PyPI downloads


`nnetsauce` does Statistical/Machine Learning (ML) using advanced combinations of randomized and quasi-randomized *neural* networks layers. It contains models for **regression**, **classification**, and **time series forecasting**. Every ML model in `nnetsauce` is based on components **g(XW + b)**, where:

- **X** is a matrix containing explanatory variables and optional clustering information. Clustering the inputs helps in taking into account data's heterogeneity before model fitting.
- **W** creates new, additional explanatory variables from **X**. **W** can be drawn from various random and quasirandom sequences.
- **b** is an optional bias parameter.
- **g** is an *activation function* such as the hyperbolic tangent or the sigmoid function, that renders the combination of explanatory variables  -- through **W** -- nonlinear. 

`nnetsauce`'s **source code** is `available on GitHub <https://github.com/thierrymoudiki/nnetsauce>`_. You can read blog posts about `nnetsauce` `here <https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN>`_, and for current references, consult section :ref:`ref-references`. 

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   Quickstart/Installation
   Quickstart/Example of use

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   APIDocumentation/Base model
   APIDocumentation/Regression models
   APIDocumentation/Classification models
   APIDocumentation/Time series models

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   Contributing/Guidelines
   Contributing/Testing

.. toctree::
   :maxdepth: 1
   :caption: Citing nnetsauce

   Citation

.. toctree::
   :maxdepth: 1
   :caption: License

   License

.. toctree::
   :maxdepth: 1
   :caption: References

   References/References

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
