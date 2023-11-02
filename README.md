nnetsauce
--------

![nnetsauce logo](the-nnetsauce.png)

<hr>

Randomized and Quasi-Randomized (neural) networks.  

![PyPI](https://img.shields.io/pypi/v/nnetsauce) [![PyPI - License](https://img.shields.io/pypi/l/nnetsauce)](https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/nnetsauce)](https://pepy.tech/project/nnetsauce) 
[![Downloads](https://anaconda.org/conda-forge/nnetsauce/badges/downloads.svg)](https://anaconda.org/conda-forge/nnetsauce)
[![HitCount](https://hits.dwyl.com/Techtonique/nnetsauce.svg?style=flat-square)](http://hits.dwyl.com/Techtonique/nnetsauce)
[![Quality](https://www.codefactor.io/repository/github/techtonique/nnetsauce/badge)](https://www.codefactor.io/repository/github/techtonique/nnetsauce)
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/nnetsauce/)


## Contents 
 [Installing for Python and R](#installing-for-Python-and-R) |
 [Package description](#package-description) |
 [Quick start](#quick-start) |
 [Contributing](#Contributing) |
 [Tests](#Tests) |
 [Dependencies](#dependencies) |
 [Citing `nnetsauce`](#Citation) |
 [API Documentation](#api-documentation) |
 [References](#References) |
 [License](#License) 


## Installing (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```

- __2nd method__: using `conda` (Linux and macOS only for now)

```bash
conda install -c conda-forge nnetsauce 
```

(Note to self or developers: https://github.com/conda-forge/nnetsauce-feedstock and https://conda-forge.org/docs/maintainer/adding_pkgs.html#step-by-step-instructions)

- __3rd method__: from Github, for the development version

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

__General rule for using the package in R__:  object accesses with `.`'s are replaced by `$`'s. R Examples can be found in the package, once installed, by typing (in R console):
```R
?nnetsauce::MultitaskClassifier
```
For a list of available models, visit [https://techtonique.github.io/nnetsauce/](https://techtonique.github.io/nnetsauce/).



## Package description

A package for Statistical/Machine Learning using Randomized and Quasi-Randomized (neural) networks. See next section. 

## Quick start

There are multiple [examples here on GitHub](https://github.com/Techtonique/nnetsauce/tree/master/examples), plus [notebooks](https://github.com/Techtonique/nnetsauce/tree/master/nnetsauce/demo) (including R Markdown notebooks). 

You can also read these [blog posts](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN).

_Deep (quasi-randomized neural) networks example_

```python
import nnetsauce as ns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from time import time



digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)


# layer 1 (base layer) ----
print(" \n layer 1 ----- \n")
layer1_clf = RandomForestClassifier(n_estimators=10, random_state=123)

start = time() 

layer1_clf.fit(X_train, y_train)

# Accuracy in layer 1
print(layer1_clf.score(X_test, y_test))


# layer 2 using layer 1 ----
print(" \n layer 2 ----- \n")
layer2_clf = ns.CustomClassifier(obj = layer1_clf, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='uniform', activation_name='relu', 
                        n_clusters=2, seed=123)
layer2_clf.fit(X_train, y_train)

# Accuracy in layer 2
print(layer2_clf.score(X_test, y_test))


# layer 3 using layer 2 ----
print(" \n layer 3 ----- \n")
layer3_clf = ns.CustomClassifier(obj = layer2_clf, n_hidden_features=10, 
                        direct_link=True, bias=True, dropout=0.7,
                        nodes_sim='uniform', activation_name='relu', 
                        n_clusters=2, seed=123)
layer3_clf.fit(X_train, y_train)

# Accuracy in layer 3
print(layer3_clf.score(X_test, y_test))

print(f"Elapsed {time() - start}") 
```

## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. If you're not comfortable with Git/Version Control yet, please use [this form](https://forms.gle/tm7dxP1jSc75puAb9) to provide a feedback.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting files: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```

A few things that we could explore are:

- Enrich the [tests](#Tests)
- Any benchmarking of `nnetsauce` models can be stored in [demo](/nnetsauce/demo) (notebooks) or [examples](./examples) (flat files), with the following naming convention:  `yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]`


## Tests

**Ultimately**, tests for `nnetsauce`'s features **will** be located [here](nnetsauce/tests). In order to run them and obtain tests' coverage (using [`nose2`](https://nose2.readthedocs.io/en/latest/)), you'll do: 

- Install packages required for testing: 

```bash
pip install nose2
pip install coverage
```

- Run tests and print coverage:

```bash
git clone https://github.com/Techtonique/nnetsauce.git
cd nnetsauce
nose2 --with-coverage
```

- Obtain coverage reports:

At the command line:

```bash
coverage report -m
```

  or an html report:

```bash
coverage html
```

## API Documentation

- [https://techtonique.github.io/nnetsauce/](https://techtonique.github.io/nnetsauce/)


## Citation

```
@misc{moudiki2019nnetsauce,
author={Moudiki, Thierry},
title={\code{nnetsauce}, {A} package for {S}tatistical/{M}achine {L}earning using {R}andomized and {Q}uasi-{R}andomized (neural) networks.,
howpublished={\url{https://github.com/thierrymoudiki/nnetsauce}},
note={BSD 3-Clause Clear License. Version 0.x.x.},
year={2019--2023}
}
```

## References

- Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at: https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters

- Moudiki, T. (2019). Multinomial logistic regression using quasi-randomized networks. Available at: https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks

- Moudiki  T,  Planchet  F,  Cousin  A  (2018).   “Multiple  Time  Series  Forecasting Using  Quasi-Randomized  Functional  Link  Neural  Networks. ”Risks, 6(1), 22. Available at: https://www.mdpi.com/2227-9091/6/1/22


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 
