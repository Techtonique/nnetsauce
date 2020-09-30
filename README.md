nnetsauce
--------

![nnetsauce logo](the-nnetsauce.png)

<hr>

Randomized and Quasi-Randomized (neural) networks.  

![PyPI](https://img.shields.io/pypi/v/nnetsauce) [![PyPI - License](https://img.shields.io/pypi/l/nnetsauce)](https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/nnetsauce)](https://pepy.tech/project/nnetsauce)
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


- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

### R 

- __1st method__: From Github, in R console:

```r
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
```

__General rule for using the package in R__:  object accesses with `.`'s are replaced by `$`'s. See also [Quick start](#quick-start).



## Package description

A package for Statistical/Machine Learning using Randomized and Quasi-Randomized (neural) networks. See next section. 

## Quick start

[https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN)

## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. If you're not comfortable with Git/Version Control yet, please use [this form](https://forms.gle/tm7dxP1jSc75puAb9) to provide a feedback.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting files: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```

A few things that we could explore are:

- Enrich the [tests](#Tests)
- Continue to make `nnetsauce` available to `R` users --> [here](./R-package)
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

## Dependencies 

- Numpy
- Scipy
- scikit-learn

## Citation

```
@misc{moudiki2019nnetsauce,
author={Moudiki, Thierry},
title={\code{nnetsauce}, {A} package for {S}tatistical/{M}achine {L}earning using {R}andomized and {Q}uasi-{R}andomized (neural) networks.,
howpublished={\url{https://github.com/thierrymoudiki/nnetsauce}},
note={BSD 3-Clause Clear License. Version 0.x.x.},
year={2019--2020}
}
```

## References

- Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at: https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters

- Moudiki, T. (2019). Multinomial logistic regression using quasi-randomized networks. Available at: https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks

- Moudiki  T,  Planchet  F,  Cousin  A  (2018).   “Multiple  Time  Series  Forecasting Using  Quasi-Randomized  Functional  Link  Neural  Networks. ”Risks, 6(1), 22. Available at: https://www.mdpi.com/2227-9091/6/1/22


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 
