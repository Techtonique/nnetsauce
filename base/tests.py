import numpy as np
from sklearn import datasets   
import nnetsauce
    
# Example 1 -----

n_features = 4
n_samples = 10
X, y = datasets.make_regression(n_features=n_features, 
                   n_samples=n_samples, 
                   random_state=0)

fit_obj = Base(n_hidden_features=3, 
             activation_name='relu', 
             n_clusters=2)

centered_y, scaled_Z = fit_obj.preproc_training_set(y=y, X=X)

print(centered_y.shape)
print(scaled_Z.shape)
print(centered_y.mean())
print(scaled_Z.mean(axis = 0))
print(np.sqrt(scaled_Z.var(axis = 0)))

fit_obj.fit(X, y) 
print(fit_obj.beta)
print(len(fit_obj.beta))
print(fit_obj.predict(X))
    
 
#    # Example 2 -----
    
diabetes = datasets.load_diabetes()

# data snippet
diabetes.feature_names

# shape 
diabetes.data.shape
diabetes.target.shape

# define X and y
X = diabetes.data 
y = diabetes.target

fit_obj = Base(n_hidden_features=3, 
                   activation_name='relu', 
                   n_clusters=2)

centered_y, scaled_Z = fit_obj.preproc_training_set(y=y, X=X)

print(centered_y.shape)
print(scaled_Z.shape)
print(centered_y.mean())
print(scaled_Z.mean(axis = 0))
print(np.sqrt(scaled_Z.var(axis = 0)))

fit_obj.fit(X, y)    
z = fit_obj.predict(X) - y
print(fit_obj.predict(X))
print(y)
print(z.mean())

