"""Model selection"""

# Authors: Thierry Moudiki
#
# License: BSD 3

# MTS -----

from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesSplit(TimeSeriesSplit):
    """Time Series cross-validator"""
    
    def __init__(self):
        # do something
        # do something
        # do something
        return self
    
    def split(self, X, initial_window=5, horizon=5, fixed_window=False):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        initial_window : int, blah
        
        horizon : int, blah
        
        fixed_window : int, blah

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # assert initial_window
        # assert horizon
        # assert fixed_window
        
        # set n_splits after (?)
        self.n_splits = 0
        
        return self







