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