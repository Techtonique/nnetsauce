# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

from .helpers import scd, sgd, one_hot_encode


class Optimizer:
    """Optimizer class

    Attributes:

        type_optim: str
            type of optimizer, (currently) either 'sgd' (stochastic minibatch gradient descent)
            or 'scd' (stochastic minibatch coordinate descent)

        num_iters: int
            number of iterations of the optimizer

        learning_rate: float
            step size

        batch_prop: float
            proportion of the initial data used at each optimization step

        learning_method: str
            "poly" - learning rate decreasing as a polynomial function
            of # of iterations (default)
            "exp" - learning rate decreasing as an exponential function
            of # of iterations
            "momentum" - gradient descent using momentum

        randomization: str
            type of randomization applied at each step
            "strat" - stratified subsampling (default)
            "shuffle" - random subsampling

        mass: float
            mass on velocity, for `method` == "momentum"

        decay: float
            coefficient of decrease of the learning rate for
            `method` == "poly" and `method` == "exp"

        tolerance: float
            early stopping parameter (convergence of loss function)

        verbose: int
            controls verbosity of gradient descent
            0 - nothing is printed
            1 - a progress bar is printed
            2 - successive loss function values are printed

    """

    # construct the object -----

    def __init__(
        self,
        type_optim="sgd",
        num_iters=100,
        learning_rate=0.01,
        batch_prop=1.0,
        learning_method="momentum",
        randomization="strat",
        mass=0.9,
        decay=0.1,
        tolerance=1e-3,
        verbose=1,
    ):
        self.type_optim = type_optim
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.batch_prop = batch_prop
        self.learning_method = learning_method
        self.randomization = randomization
        self.mass = mass
        self.decay = decay
        self.tolerance = tolerance
        self.verbose = verbose
        self.opt = None

    def fit(self, loss_func, response, x0, **kwargs):
        """Fit GLM model to training data (X, y).

        Args:

            loss_func: loss function

            response: array-like, shape = [n_samples]
            target variable (used for subsampling)

            x0: array-like, shape = [n_features]
                initial value provided to the optimizer

            **kwargs: additional parameters to be passed to
                    loss function

        Returns:

            self: object

        """

        if self.type_optim == "scd":
            self.results = scd(
                loss_func,
                response=response,
                x=x0,
                num_iters=self.num_iters,
                batch_prop=self.batch_prop,
                learning_rate=self.learning_rate,
                learning_method=self.learning_method,
                mass=self.mass,
                decay=self.decay,
                randomization=self.randomization,
                tolerance=self.tolerance,
                verbose=self.verbose,
                **kwargs
            )

        if self.type_optim == "sgd":
            self.results = sgd(
                loss_func,
                response=response,
                x=x0,
                num_iters=self.num_iters,
                batch_prop=self.batch_prop,
                learning_rate=self.learning_rate,
                learning_method=self.learning_method,
                mass=self.mass,
                decay=self.decay,
                randomization=self.randomization,
                tolerance=self.tolerance,
                verbose=self.verbose,
                **kwargs
            )

        return self

    def one_hot_encode(self, y, n_classes):
        return one_hot_encode(y, n_classes)
