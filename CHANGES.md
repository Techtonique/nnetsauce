# version 0.12.2

- Fix error in simulation (base.py)

# version 0.12.1

- Begin residuals simulation

# version 0.11.5

- Avoid division by zero in scaling

# version 0.11.4

- less dependencies in setup 

# version 0.11.0

- Implement RandomBagRegressor
- Use of a DataFrame in MTS

# version 0.10.0

- rename attributes with underscore
- add more examples to documentation

# version 0.9.6

- Fix numbers' simulations

# version 0.9.4

- Remove memoize from Simulator

# version 0.9.2

- loosen the range of Python packages versions

# version 0.9.0

- Add Poisson and Laplace regressions to GLMRegressor
- Remove smoothing weights from MTS

# version 0.8.0

- Use C++ for simulation
- Fix R Engine problem

# version 0.7.0

- RandomBag classifier cythonized

# version 0.6.0

- Documentation with MkDocs
- Cython-ready

# version 0.5.0

- contains a refactorized code for the [`Base`](https://github.com/thierrymoudiki/nnetsauce/nnetsauce/base/base.py) class, 
and for many other utilities.
- makes use of [randtoolbox](https://cran.r-project.org/web/packages/randtoolbox/index.html) 
for a faster, more scalable generation of quasi-random numbers.
- contains __a (work in progress) implementation of most algorithms on GPUs__,
 using [JAX](https://github.com/google/jax). Most of the nnetsauce's changes 
 related to GPUs are currently made on potentially time consuming operations 
 such as matrices multiplications and matrices inversions.

# version 0.4.0

- (Work in progress) documentation in `/docs`
- `MultitaskClassifier`
- Rename `Mtask` to `Multitask`
- Rename `Ridge2ClassifierMtask` to `Ridge2MultitaskClassifier`


# version 0.3.3

- Use "return_std" only in predict for MTS object


# version 0.3.2

- Fix for potential error "Sample weights must be 1D array or scalar"


# version 0.3.1

- One-hot encoding not cached (caused errs on multitask ridge2 classifier)
- Rename ridge to ridge2 (2 shrinkage params compared to ridge)


# version 0.3.0

- Implement ridge2 (regressor and classifier)
- Upper bound on Adaboost error
- Test Time series split


# version 0.2.0

- Add AdaBoost classifier 
- Add RandomBag classifier (bagging)
- Add multinomial logit Ridge classifier 
- Remove dependency to package `sobol_seq`(not used)


# version 0.1.0

- Initial version