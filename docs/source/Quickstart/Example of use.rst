
Example of use
''''''''''''''

In these examples, we use `nnetsauce` classes  ``CustomRegressor`` and ``CustomClassifier`` in Python, to create new models with **one hidden layer** and **three hidden layers**. There are more examples `in these notebooks <https://github.com/thierrymoudiki/nnetsauce/tree/master/nnetsauce/demo>`_ or `this directory <https://github.com/thierrymoudiki/nnetsauce/tree/master/examples>`_; feel free to contribute yours in there too. For **R examples**, you can type the following commands in R console: 

.. code-block:: r

	# CustomRegressor is a nnetsauce class
	help("CustomRegressor")


And read the **Examples** section in the Help page displayed. Any other class name from the API Documentation (see sections :ref:`ref-regression-models`, :ref:`ref-classification-models` and :ref:`ref-time-series-models`) can be used instead of ``CustomRegressor``. 


Example 1 (one layer)
---------------------

In this **first example**, the models created all have one layer, and  are respectively based on Bayesian Ridge regression, Elastic Net regression, and a Gaussian Process Classifier. We start by installing `sklearn` at the command line -- though `nnetsauce` will work with any object having methods ``fit`` and ``predict`` : 

.. code-block:: console

    pip install sklearn

Now, we can use `nnetsauce` to add **one hidden layer** to these models: 

.. code-block:: python

	import nnetsauce as ns
	import numpy as np      
	import matplotlib.pyplot as plt

	from sklearn import datasets, metrics
	from sklearn import linear_model, gaussian_process

	# load datasets
	diabetes = datasets.load_diabetes()
	X = diabetes.data 
	y = diabetes.target

	breast_cancer = datasets.load_breast_cancer()
	Z = breast_cancer.data
	t = breast_cancer.target

	# Base models for nnetsauce's Custom class
	regr = linear_model.BayesianRidge()
	regr2 = linear_model.ElasticNet()
	regr3 = gaussian_process.GaussianProcessClassifier()

	# create object Custom 
	fit_obj = ns.CustomRegressor(obj=regr, n_hidden_features=100, 
	                    direct_link=False, bias=True,
	                    activation_name='tanh', n_clusters=2)

	fit_obj2 = ns.CustomRegressor(obj=regr2, n_hidden_features=5, 
	                    direct_link=True, bias=False,
	                    activation_name='relu', n_clusters=0)

	fit_obj3 = ns.CustomClassifier(obj = regr3, n_hidden_features=5, 
	                    direct_link=True, bias=True,
	                    activation_name='relu', n_clusters=2)

	# fit model 1 on training set
	index_train_1 = range(350)
	index_test_1 = range(350, 442)
	fit_obj.fit(X[index_train_1,:], y[index_train_1])

	# predict on test set 
	print(fit_obj.predict(X[index_test_1,:]))

	# fit model 2 on training set
	fit_obj2.fit(X[index_train_1,:], y[index_train_1])

	# predict on test set 
	print(fit_obj2.predict(X[index_test_1,:]))

	# fit model 3 on training set
	index_train_2 = range(455)
	index_test_2 = range(455, 569)
	fit_obj3.fit(Z[index_train_2,:], t[index_train_2])

	# accuracy on test set 
	print(fit_obj3.score(Z[index_test_2,:], t[index_test_2]))


Example 2 (three layers)
------------------------

In this **second example**, the model created has **three hidden layers** and is based on Bayesian Ridge regression: 

.. code-block:: python

	index_train = range(100)
	index_test = range(100, 125)

	# layer 1 (base layer) ----
	layer1_regr = linear_model.BayesianRidge()
	layer1_regr.fit(X[index_train,:], y[index_train])

	# RMSE score on test set
	print(np.sqrt(metrics.mean_squared_error(y[index_test], layer1_regr.predict(X[index_test,:]))))


	# layer 2 using layer 1 ----
	layer2_regr = ns.CustomRegressor(obj = layer1_regr, n_hidden_features=3, 
	                        direct_link=True, bias=True, 
	                        nodes_sim='sobol', activation_name='tanh', 
	                        n_clusters=2)
	layer2_regr.fit(X[index_train,:], y[index_train])

	# RMSE score on test set
	print(np.sqrt(layer2_regr.score(X[index_test,:], y[index_test])))

	# layer 3 using layer 2 ----
	layer3_regr = ns.CustomRegressor(obj = layer2_regr, n_hidden_features=5, 
	                        direct_link=True, bias=True, 
	                        nodes_sim='hammersley', activation_name='sigmoid', 
	                        n_clusters=2)
	layer3_regr.fit(X[index_test,:], y[index_test])

	# RMSE score on test set
	print(np.sqrt(layer3_regr.score(X[index_test,:], y[index_test])))

The entire **API documentation is presented in the next section**. You can also refer to Indices and tables in the homepage, or use the search feature. 