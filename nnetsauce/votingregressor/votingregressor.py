import numpy as np
from sklearn.ensemble import VotingRegressor

class MedianVotingRegressor(VotingRegressor):
    def predict(self, X):
        """
        Predict using the median of the base regressors' predictions.
        
        Parameters:
        X (array-like): Feature matrix for predictions.
        
        Returns:
        y_pred (array): Median of predictions from the base regressors.
        """
        predictions = np.asarray([regressor.predict(X) for regressor in self.estimators_])
        return np.median(predictions, axis=0)