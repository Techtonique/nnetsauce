
# version 0.20.5

- Import `all_estimators` from `sklearn.utils` 

# version 0.18.0

- Bayesian `CustomRegressor`
- Conformalized `CustomRegressor` (`splitconformal` and `localconformal` for now)
- See [this example](./examples/conformal_preds.py), [this example](./examples/custom_bayesian_regression.py), and [this notebook](./nnetsauce/demo/thierrymoudiki_20240317_conformal_regression.ipynb)

# version 0.17.2

- `self.n_classes_ = len(np.unique(y))` # for compatibility with sklearn 

# version 0.17.1

- `preprocess`ing for all `LazyDeep*`

# version 0.17.0

- Attribute `estimators` (a list of `Estimator`'s as strings) for `LazyClassifier`, 
  `LazyRegressor`, `LazyDeepClassifier`, `LazyDeepRegressor`, `LazyMTS`, and `LazyDeepMTS`
- New documentation for the package, using `pdoc` (not `pdoc3`)
- Remove external regressors `xreg` at inference time for `MTS` and `DeepMTS`
- New class `Downloader`: querying the R universe API for datasets (see 
  https://thierrymoudiki.github.io/blog/2023/12/25/python/r/misc/mlsauce/runiverse-api2 for similar example in `mlsauce`)
- Add custom metric to `Lazy*`
- Rename Deep regressors and classifiers to `Deep*` in `Lazy*`
- Add attribute `sort_by` to `Lazy*` -- sort the data frame output by a given metric
- Add attribute `classes_` to classifiers (ensure consistency with sklearn)

# version 0.16.8

- Subsample response by using the **number of rows**, not only a percentage (see [https://thierrymoudiki.github.io/blog/2024/01/22/python/nnetsauce-subsampling](https://thierrymoudiki.github.io/blog/2024/01/22/python/nnetsauce-subsampling))
- Improve consistency with sklearn's v1.2, for `OneHotEncoder`

# version 0.16.3

- add **robust scaler** 
- relatively **faster scaling** in preprocessing
- **Regression-based classifiers** (see [https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers](https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers))
- `DeepMTS` (multivariate time series forecasting with deep quasi-random layers): see https://thierrymoudiki.github.io/blog/2024/01/15/python/quasirandomizednn/forecasting/DeepMTS
- AutoML for `MTS` (multivariate time series forecasting): see https://thierrymoudiki.github.io/blog/2023/10/29/python/quasirandomizednn/MTS-LazyPredict
- AutoML for `DeepMTS` (multivariate time series forecasting): see https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20240106_LazyDeepMTS.ipynb
- Spaghetti plots for `MTS` and `DeepMTS` (multivariate time series forecasting): see https://thierrymoudiki.github.io/blog/2024/01/15/python/quasirandomizednn/forecasting/DeepMTS
- Subsample continuous and discrete responses  

# version 0.16.0

- actually implement _deep_ `Estimator`s in [`/deep`](./nnetsauce/deep/) (in addition to [`/lazypredict`](./nnetsauce/lazypredict/))
- include new multi-output regression-based classifiers (see [https://thierrymoudiki.github.io/blog/2021/09/26/python/quasirandomizednn/classification-using-regression](https://thierrymoudiki.github.io/blog/2021/09/26/python/quasirandomizednn/classification-using-regression) for more details)
- use proper names for `Estimator`s in [`/lazypredict`](./nnetsauce/lazypredict/) and [`/deep`](./nnetsauce/deep/)
- expose `SubSampler` (stratified subsampling) to the external API 

# version 0.15.0

- lazy predict for classification and regression (see https://thierrymoudiki.github.io/blog/2023/10/22/python/quasirandomizednn/nnetsauce-lazy-predict-preview)
- lazy predict for multivariate time series (see https://thierrymoudiki.github.io/blog/2023/10/29/python/quasirandomizednn/MTS-LazyPredict)
- lazy predict for deep classifiers and regressors (see [this example for classification](./examples/lazy_custom_deep_classification.py) and [this example for regression](./examples/lazy_custom_deep_regression.py)) 

# version 0.14.0

- update and align as much as possible with R version 
- colored graphics for class MTS 

# version 0.13.0

- Fix error in nodes' simulation (base.py)
- Use residuals and KDE for predictive simulations
- `plot` method for MTS objects 

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