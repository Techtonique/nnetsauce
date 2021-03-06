

# nnetsauce | <a class="github-button" href="https://github.com/Techtonique/nnetsauce/stargazers" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" aria-label="Star nnetsauce/nnetsauce on GitHub">Star</a>

![PyPI](https://img.shields.io/pypi/v/nnetsauce) [![PyPI - License](https://img.shields.io/pypi/l/nnetsauce)](https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/nnetsauce)](https://pepy.tech/project/nnetsauce) [![Last Commit](https://img.shields.io/github/last-commit/Techtonique/nnetsauce)](https://github.com/Techtonique/nnetsauce)

Welcome to __nnetsauce__'s website.

_nnetsauce_ does Statistical/Machine Learning (ML) using advanced combinations of randomized and quasi-randomized neural networks layers. It contains models for regression, classification, and time series forecasting. Every ML model in nnetsauce is based on components g(XW + b), where:

   - X is a matrix containing explanatory variables and optional clustering information. Clustering the inputs helps in taking into account data’s heterogeneity before model fitting.
   - W creates new, additional explanatory variables from X. W can be drawn from various random and quasirandom sequences.
   - b is an optional bias parameter.
   - g is an activation function such as the hyperbolic tangent or the sigmoid function, that renders the combination of explanatory variables – through W – nonlinear.

__nnetsauce__’s source code is [available on GitHub](https://github.com/Techtonique/nnetsauce). 

You can read posts about nnetsauce [in this blog](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN), and for current references, feel free consult the section: [References](REFERENCES.md).

Looking for a specific function? You can also use the __search__ function available in the navigation bar.

## Installing (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```


- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

or 

```bash
git clone https://github.com/Techtonique/nnetsauce.git
cd nnetsauce
make install
```


### R 

- __1st method__: From Github, in R console:

```r
library(devtools)
devtools::install_github("Techtonique/nnetsauce/R-package")
library(nnetsauce)
```

__General rule for using the package in R__:  object accesses with `.`'s are replaced by `$`'s. See also [Quick start](#quick-start).


## Quickstart 

Examples of use: 

- For [classification](examples/classification.md)

- For [regression](examples/regression.md)

- For [time series](examples/time_series_examples.md)

- R examples can be found in these notebooks: 
   - [thierrymoudiki_060320_RandomBagClassifier.Rmd](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_060320_RandomBagClassifier.Rmd)
   - [thierrymoudiki_060320_Ridge2Classifier.Rmd](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_060320_Ridge2Classifier.Rmd)

## Documentation

The documentation for each model can be found (work in progress) here:

- For [classifiers](documentation/classifiers.md)

- For [regressors](documentation/regressors.md)

- For [time series](documentation/time_series.md) models


## Contributing

Want to contribute to __nnetsauce__'s development on Github, [read this](CONTRIBUTING.md)!

<script async defer src="https://buttons.github.io/buttons.js"></script>