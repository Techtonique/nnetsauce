nnetsauce
--------

![nnetsauce logo](the-nnetsauce.png)

<hr>

Randomized and Quasi-Randomized (neural) networks. 

![PyPI](https://img.shields.io/pypi/v/nnetsauce) [![PyPI - License](https://img.shields.io/pypi/l/nnetsauce)](https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/nnetsauce)](https://pepy.tech/project/nnetsauce) 
[![Downloads](https://anaconda.org/conda-forge/nnetsauce/badges/downloads.svg)](https://anaconda.org/conda-forge/nnetsauce)
[![HitCount](https://hits.dwyl.com/Techtonique/nnetsauce.svg?style=flat-square)](http://hits.dwyl.com/Techtonique/nnetsauce)
[![CodeFactor](https://www.codefactor.io/repository/github/techtonique/nnetsauce/badge)](https://www.codefactor.io/repository/github/techtonique/nnetsauce)
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

(Note to self or developers: https://github.com/conda-forge/nnetsauce-feedstock and https://conda-forge.org/docs/maintainer/adding_pkgs.html#step-by-step-instructions and https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/#the-whole-ci-cd-workflow)

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

**From GitHub**

```bash
remotes::install_github("Techtonique/nnetsauce_r") # the repo is in this organization
```

**From R-universe**

```bash
install.packages('nnetsauce', repos = c('https://techtonique.r-universe.dev',
'https://cloud.r-project.org'))
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

_Lazy Deep (quasi-randomized neural) networks example_

```python
!pip install nnetsauce --upgrade
```

```python
import os
import nnetsauce as ns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from time import time

data = load_breast_cancer()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
start = time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(f"\n\n Elapsed: {time()-start} seconds \n")

model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)

display(models)
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
- Any benchmarking of `nnetsauce` models can be stored in [demo](/nnetsauce/demo) (notebooks) or [examples](./examples) (flat files), with the following naming convention:  `yourgithubname_yyyymmdd_shortdescriptionofdemo.[py|ipynb|R|Rmd]`


## Tests

**Ultimately**, tests for `nnetsauce`'s features **will** be located [here](nnetsauce/tests). In order to run them and obtain tests' coverage (using [`nose2`](https://nose2.readthedocs.io/en/latest/)), you'll do: 

- Install packages required for testing: 

```bash
pip install nose2
pip install coverage
```

- Run tests (in cloned repo) and print coverage:

```bash
make run-tests
make coverage
```

## API Documentation

- [https://techtonique.github.io/nnetsauce/](https://techtonique.github.io/nnetsauce/)


## Citation (BibTeX entry)

Replace `Version x.x.x` by the version number you've used. 

```
@misc{moudiki2019nnetsauce,
author={Moudiki, T.},
title={nnetsauce, {A} package for {S}tatistical/{M}achine {L}earning using {R}andomized and {Q}uasi-{R}andomized (neural) networks.},
howpublished={\url{https://github.com/thierrymoudiki/nnetsauce}},
note={BSD 3-Clause Clear License. Version x.x.x},
year={2019--2024}
}}
```

## References

- Probabilistic Forecasting with nnetsauce (using Density Estimation, Bayesian inference, Conformal prediction and Vine copulas): nnetsauce presentation at sktime meetup (2024) https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas
  
- Probabilistic Forecasting with Randomized/Quasi-Randomized networks at the International Symposium on Forecasting (2024) https://www.researchgate.net/publication/381957724_Probabilistic_Forecasting_with_RandomizedQuasi-Randomized_networks_presentation_at_the_International_Symposium_on_Forecasting_2024

- Moudiki, T. (2024). Regression-based machine learning classifiers. Available at: https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers

- Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at: https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters

- Moudiki, T. (2019). Multinomial logistic regression using quasi-randomized networks. Available at: https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks

- Moudiki  T,  Planchet  F,  Cousin  A  (2018).   “Multiple  Time  Series  Forecasting Using  Quasi-Randomized  Functional  Link  Neural  Networks. ”Risks, 6(1), 22. Available at: https://www.mdpi.com/2227-9091/6/1/22


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 
