nnetsauce's documentation
=========================================

`nnetsauce` does Statistical/Machine Learning (ML) using advanced combinations of randomized and quasi-randomized *neural* networks layers. It contains models for **regression**, **classification**, and **time series forecasting**. Every ML model in `nnetsauce` is based on components **g(XW + b)**, where:

- **X** is a matrix containing explanatory variables and optional clustering information. Clustering the inputs helps in taking into account data's heterogeneity before model fitting.
- **W** creates new, additional explanatory variables from **X**. **W** can be drawn from various random and quasirandom sequences.
- **b** is an optional bias parameter.
- **g** is an *activation function* such as the hyperbolic tangent or the sigmoid function, that renders the combination of explanatory variables  -- through **W** -- nonlinear. 

`nnetsauce`'s **source code** is `available on GitHub <https://github.com/thierrymoudiki/nnetsauce>`_. 

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   Quickstart/Installation
   Quickstart/Example of use

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   APIDocumentation/Regression models
   APIDocumentation/Classification models
   APIDocumentation/Time series models

.. toctree::
   :maxdepth: 1
   :caption: Contributing

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
