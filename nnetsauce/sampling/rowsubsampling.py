from .helpers import dosubsample

class SubSampler:
    """Subsampling class.

    Attributes:

       y: array-like, shape = [n_samples]
           Target values.

       row_sample: double
           subsampling fraction

       seed: int
           reproductibility seed
    
    Returns:

        indices of subsampled y

    """
    def __init__(self, y, row_sample=0.8, seed=123):
        self.y = y
        self.row_sample = row_sample
        self.seed = seed
        self.indices = None 

    def subsample(self):
        self.indices = dosubsample(self.y, self.row_sample, self.seed)
        return self.indices
