from . import _rowsubsamplingc as rowsubsamplingc


class SubSampler:
    """Subsampling class.

    Attributes:

       y: array-like, shape = [n_samples]
           Target values.

       row_sample: double
           subsampling fraction

       seed: int
           reproductibility seed

    """

    def __init__(self, y, row_sample=0.8, seed=123):
        self.y = y
        self.row_sample = row_sample
        self.seed = seed

    def subsample(self):
        return rowsubsamplingc.subsamplec(self.y, self.row_sample, self.seed)
