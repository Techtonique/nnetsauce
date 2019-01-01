# nnetsauce 

This package does Machine Learning by using various (advanced) combinations of single layer neural networks. Every model in `nnetsauce` is based on the component $g(XW + b)$, where:

- $X$ is a matrix containing the explanatory variables 
- $W$ constructs the nodes in the hidden layer from $X$
- $b$ is an optional bias parameter
- $g$ is an activation function such as the hyperbolic tangent (among others)  

### Description

Currently, $3$ models are implemented in the package. If your response variable is $y$ ,then:

- model `Base` adjusts a linear model to y, as a function of $X$ and $g(XW + b)$  
- model `BayesianRVFL` adds a regularization parameter to model `Base`, which prevents overfitting 
- model `BayesianRVFL2` adds $2$ regularization parameters to model `Base`, which prevents overfitting
- model `Custom` works with any object `obj` possessing the methods `obj.fit()` and `obj.predict()`. Notably, the model can be applied to any `scikit-learn` model. 


### Examples

Examples of use.

Examples of use with sklearn.

Other examples can be found in file 

## Installation 

- __1st method__: from Github

```
git clone https://github.com/thierrymoudiki/nnetsauce.git
cd nnetsauce
python setup.py install
```

- __2nd method__: by using `pip`

```
TODO
```



## Dependencies 

- Numpy
- Scipy
- scikit-learn


## References

- Ref1
- Ref2
- Ref3

