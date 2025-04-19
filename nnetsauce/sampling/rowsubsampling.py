import numpy as np
from .helpers import dosubsample


class SubSampler:
    """Subsampling class.

    Attributes:

       y: array-like, shape = [n_samples]
           Target values.

       row_sample: double
           subsampling fraction

       n_samples: int
            subsampling by using the number of rows (supersedes row_sample)

       seed: int
           reproductibility seed

       n_jobs: int
            number of jobs to run in parallel

       verbose: bool
            print progress messages and bars
    """

    def __init__(
        self,
        y,
        row_sample=0.8,
        n_samples=None,
        seed=123,
        n_jobs=None,
        verbose=False,
    ):
        self.y = y
        self.n_samples = n_samples
        if self.n_samples is None:
            assert (
                row_sample < 1 and row_sample >= 0
            ), "'row_sample' must be provided, plus < 1 and >= 0"
            self.row_sample = row_sample
        else:
            assert self.n_samples < len(y), "'n_samples' must be < len(y)"
            self.row_sample = self.n_samples / len(y)
        self.seed = seed
        self.indices = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def subsample(self):
        """Returns indices of subsampled input data.

        Examples:

        <ul>
            <li> <a href="https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20240105_subsampling.ipynb">20240105_subsampling.ipynb</a> </li>
            <li> <a href="https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20240131_subsampling_nsamples.ipynb">20240131_subsampling_nsamples.ipynb</a> </li>
        </ul>

        """
        self.indices = dosubsample(
            y=self.y,
            row_sample=self.row_sample,
            seed=self.seed,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        return self.indices
