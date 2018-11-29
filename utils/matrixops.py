import numpy as np


# column bind
def cbind(x, y):
    
    #if len(x.shape) == 1 or len(y.shape) == 1: 
        return np.column_stack((x, y))
    #else:
    #    return np.concatenate((x, y), 
    #                          axis = 1)
        

# computes t(x)%*%y
def crossprod(x, y = None):
    # assert on dimensions
    if (y is None):
        return np.dot(x.transpose(), x)
    else:
        return np.dot(x.transpose(), y)


# one-hot encoding 
def one_hot_encode(x_clusters, n_clusters):
    
    assert max(x_clusters) <= n_clusters
    
    n_obs = len(x_clusters)
    res = np.zeros((n_obs, n_clusters))
    
    for i in range(n_obs):
        res[i, x_clusters[i]] = 1
            
    return res 


# row bind
def rbind(x, y):
    
    #if len(x.shape) == 1 or len(y.shape) == 1: 
        return np.row_stack((x, y))
    #else: 
    #    return np.concatenate((x, y), 
    #                      axis = 0)

    
# scale matrix
def scale_matrix(X, x_means=None, x_vars=None):
    
    if ((x_means is None) & (x_vars is None)):
        x_means = X.mean(axis = 0)
        x_vars = X.var(axis = 0)
        return ((X - x_means)/np.sqrt(x_vars), 
                x_means, 
                x_vars)
    
    if ((x_means is not None) & (x_vars is None)):
        return X - x_means
    
    if ((x_means is None) & (x_vars is not None)):
        return X/np.sqrt(x_vars)
    
    if ((x_means is not None) & (x_vars is not None)):
        return (X - x_means)/np.sqrt(x_vars)


# computes x%*%t(y)
def tcrossprod(x, y = None):
    # assert on dimensions
    if (y is None):
        return np.dot(x, x.transpose())   
    else: 
        return np.dot(x, y.transpose()) 


# convert vector to numpy array
def to_np_array(X):
    return np.array(X.copy(), ndmin=2)
