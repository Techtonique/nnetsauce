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
        
       n_jobs: int
            number of jobs to run in parallel
        
       verbose: bool
            print progress messages and bars

    Returns:

        indices of subsampled y

    """

    def __init__(self, y, row_sample=0.8, seed=123, n_jobs=None, verbose=False):
        self.y = y
        self.row_sample = row_sample
        self.seed = seed
        self.indices = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def subsample(self):
        self.indices = dosubsample(
            self.y, self.row_sample, self.seed, self.n_jobs, self.verbose
        )
        return self.indices
