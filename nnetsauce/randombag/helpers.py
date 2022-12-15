# Authors: Thierry Moudiki
#
# License: BSD 3

import functools
import pickle
import numpy as np

from numpy.linalg import lstsq
from numpy.linalg import norm
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
from ..utils import Progbar


# 0 - utils -----

# 1 main fitting loop -----       

# For classification
def rbagloop_classification(base_learner, X, y, n_estimators, verbose, seed):

    voter = {}    

    if verbose == 1:

        pbar = Progbar(n_estimators)

        for m in range(n_estimators):   
            
            try:

                base_learner.set_params(seed=seed + m * 1000)

                base_learner.fit(np.asarray(X), np.asarray(y))
                
                voter[m] = pickle.loads(pickle.dumps(base_learner, -1))                                

                pbar.update(m)

            except:

                pbar.update(m)

                continue
    
        pbar.update(n_estimators)
    
        return voter

    # verbose != 1:
    for m in range(n_estimators):   
        
        try:

            base_learner.set_params(seed=seed + m * 1000)
                
            base_learner.fit(np.asarray(X), np.asarray(y))

            voter[m] = pickle.loads(pickle.dumps(base_learner, -1))                

        except:            

            continue

    return voter

# For regression
def rbagloop_regression(base_learner, X, y, n_estimators, verbose, seed):

    voter = {}    

    if verbose == 1:

        pbar = Progbar(n_estimators)

        for m in range(n_estimators):   
            
            try:

                base_learner.set_params(seed=seed + m * 1000)

                base_learner.fit(np.asarray(X), np.asarray(y))
                
                voter[m] = pickle.loads(pickle.dumps(base_learner, -1))                

                pbar.update(m)

            except:

                pbar.update(m)

                continue
    
        pbar.update(n_estimators)
    
        return voter

    # verbose != 1:
    for m in range(n_estimators):   
        
        try:
            base_learner.set_params(seed=seed + m * 1000)
                
            base_learner.fit(np.asarray(X), np.asarray(y))

            voter[m] = pickle.loads(pickle.dumps(base_learner, -1))                         

        except:            

            continue

    return voter


