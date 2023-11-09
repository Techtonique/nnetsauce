# Adapted from Xu et al. @ ICML 2021 and 2023
from time import time
import pandas as pd
import numpy as np

from .utils_SPCI import (
    binning,
    generate_bootstrap_samples,
    strided_app,
)

# Only used:
# SPCI_and_EnbPI
# fit_bootstrap_models_online_multistep
# compute_PIs_Ensemble_online
# Attributes
# Ensemble_pred_interval_centers: point forecast
# PIs_Ensemble: predictions intervals PIs_Ensemble['lower'], PIs_Ensemble['upper']

#### Main Class ####
class SPCI_and_EnbPI:
    """
        Create prediction intervals assuming Y_t = f(X_t) + \sigma(X_t)\eps_t
        Currently, assume the regression function is by default MLP implemented with PyTorch, as it needs to estimate BOTH f(X_t) and \sigma(X_t), where the latter is impossible to estimate using scikit-learn modules

        Most things carry out, except that we need to have different estimators for f and \sigma.

        fit_func = None: use MLP above
    """

    def __init__(self, X_train, Y_train, h=5, fit_func=None):

        self.regressor = fit_func
        self.X_train = X_train
        self.Y_train = Y_train

        # Predicted training data centers by EnbPI
        n, n1 = self.X_train.shape[0], h
        self.Ensemble_train_interval_centers = np.ones(n) * np.inf
        self.Ensemble_train_interval_sigma = np.ones(n) * np.inf
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = np.ones(n1) * np.inf
        self.Ensemble_pred_interval_sigma = np.ones(n1) * np.inf
        self.Ensemble_online_resid = np.ones(n + n1) * np.inf  # LOO scores
        self.beta_hat_bins = []

        #### Other hyperparameters for training (mostly simulation) ####
        # QRF training & how it treats the samples
        self.weigh_residuals = False  # Whether we weigh current residuals more.
        self.c = (
            0.995  # If self.weight_residuals, weights[s] = self.c ** s, s\geq 0
        )
        self.n_estimators = 10  # Num trees for QRF
        self.max_d = 2  # Max depth for fitting QRF
        self.criterion = "mse"  # 'mse' or 'mae'
        # search of \beta^* \in [0,\alpha]
        self.bins = 5  # break [0,\alpha] into bins
        # how many LOO training residuals to use for training current QRF
        self.T1 = None  # None = use all

        # additional attributes
        self.d = None
        self.fit_sigmaX = None
        self.alpha = None
        self.past_window = None
        self.train_idx = None
        self.b = None
        self.PIs_Ensemble = None
        self.test_idx = None
        self.h = None

    def one_boot_prediction(self, X_boot, Y_boot, X_full):
        # NOTE, NO sigma estimation because these methods by deFAULT are fitting Y, but we have no observation of errors
        model = self.regressor
        model.fit(X_boot, Y_boot)
        boot_fX_pred = model.predict(X_full, h=self.h)
        boot_sigma_pred = 0
        return boot_fX_pred, boot_sigma_pred

    def fit_bootstrap_models_online_multistep(
        self, B=100, fit_sigmaX=True, stride=1, h=5
    ):
        """
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
          fit_sigmaX: If False, just avoid predicting \sigma(X_t) by defaulting it to 1

          stride: int. If > 1, then we perform multi-step prediction, where we have to fit stride*B boostrap predictors.
            Idea: train on (X_i,Y_i), i=1,...,n-stride
            Then predict on X_1,X_{1+s},...,X_{1+k*s} where 1+k*s <= n+n1
            Note, when doing LOO prediction thus only care above the points above

          h: int. forecasting horizon
        """
        n, self.d = self.X_train.shape
        self.fit_sigmaX = fit_sigmaX
        self.h = h
        n1 = h  # forecasting horizon
        N = n - stride + 1  # Total training data each one-step predictor sees
        # We make prediction every s step ahead, so these are feature the model sees
        train_pred_idx = np.arange(0, n, stride)
        # We make prediction every s step ahead, so these are feature the model sees
        test_pred_idx = np.arange(n, n + n1, stride)
        self.train_idx = train_pred_idx
        self.test_idx = test_pred_idx
        # Only contains features that are observed every stride steps
        X_full = self.X_train[train_pred_idx, :]
        nsub, n1sub = len(train_pred_idx), len(test_pred_idx)
        for s in range(stride):
            # 1 - Create containers for predictions
            # hold indices of training data for each f^b
            boot_samples_idx = generate_bootstrap_samples(N, N, B)
            # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
            in_boot_sample = np.zeros((B, N), dtype=bool)
            # hold predictions from each f^b for fX and sigma&b for sigma
            boot_predictionsFX = np.zeros((B, nsub + n1sub))
            boot_predictionsSigmaX = np.ones((B, nsub + n1sub))
            # We actually would just use n1sub rows, as we only observe this number of features
            out_sample_predictFX = np.zeros((n, n1sub))
            out_sample_predictSigmaX = np.ones((n, n1sub))

            # 2 - Start bootstrap prediction
            start = time()
            for b in range(B):
                self.b = b
                X_boot, Y_boot = (
                    self.X_train[boot_samples_idx[b], :],
                    self.Y_train[s : s + N][boot_samples_idx[b],],
                )
                in_boot_sample[b, boot_samples_idx[b]] = True
                boot_fX_pred, boot_sigma_pred = self.one_boot_prediction(
                    X_boot, Y_boot, X_full
                )
                boot_predictionsFX[b] = boot_fX_pred
                if self.fit_sigmaX:
                    boot_predictionsSigmaX[b] = boot_sigma_pred
            print(
                f"{s+1}/{stride} multi-step: finish Fitting {B} Bootstrap models, took {time()-start} secs."
            )

            # 3 - Obtain LOO residuals (train and test) and prediction for test data '''
            start = time()
            # Consider LOO, but here ONLY for the indices being predicted
            for j, i in enumerate(train_pred_idx):
                # j: counter and i: actual index X_{0+j*stride}
                if i < N:
                    b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        # All bootstrap estimators are trained on this model
                        b_keep = 0  # More rigorously, it should be None, but in practice, the difference is minor
                else:
                    # This feature is not used in training, but used in prediction
                    b_keep = range(B)
                pred_iFX = boot_predictionsFX[b_keep, j].mean()
                pred_iSigmaX = boot_predictionsSigmaX[b_keep, j].mean()
                pred_testFX = boot_predictionsFX[b_keep, nsub:].mean(0)
                pred_testSigmaX = boot_predictionsSigmaX[b_keep, nsub:].mean(0)
                # Populate the training prediction
                # We add s because of multi-step procedure, so f(X_t) is for Y_t+s
                true_idx = min(i + s, n - 1)
                self.Ensemble_train_interval_centers[true_idx] = pred_iFX
                self.Ensemble_train_interval_sigma[true_idx] = pred_iSigmaX
                resid_LOO = (self.Y_train[true_idx] - pred_iFX) / pred_iSigmaX
                out_sample_predictFX[i] = pred_testFX
                out_sample_predictSigmaX[i] = pred_testSigmaX
                self.Ensemble_online_resid[true_idx] = resid_LOO.item()
            sorted_out_sample_predictFX = out_sample_predictFX[
                train_pred_idx
            ].mean(
                0
            )  # length ceil(n1/stride)
            sorted_out_sample_predictSigmaX = out_sample_predictSigmaX[
                train_pred_idx
            ].mean(
                0
            )  # length ceil(n1/stride)
            pred_idx = np.minimum(test_pred_idx - n + s, n1 - 1)
            self.Ensemble_pred_interval_centers[
                pred_idx
            ] = sorted_out_sample_predictFX
            self.Ensemble_pred_interval_sigma[
                pred_idx
            ] = sorted_out_sample_predictSigmaX
            # pred_full_idx = np.minimum(test_pred_idx + s, n + n1 - 1)
            # resid_out_sample = (self.Y_predict[pred_idx] - sorted_out_sample_predictFX) / sorted_out_sample_predictSigmaX
            # self.Ensemble_online_resid[pred_full_idx] = resid_out_sample
        # Sanity check
        # num_inf = (self.Ensemble_online_resid == np.inf).sum()
        # if num_inf > 0:
        #     # print(
        #     #    f'Something can be wrong, as {num_inf}/{n+n1} residuals are not all computed')
        #     print(
        #         f"Something can be wrong, as {num_inf}/{n} residuals are not all computed"
        #     )
        #     print(np.where(self.Ensemble_online_resid == np.inf))

    def compute_PIs_Ensemble_online(
        self,
        alpha,
        stride=1,
        smallT=True,
        past_window=100,
        use_SPCI=False,
        quantile_regr="RF",
    ):
        """
            stride: control how many steps we predict ahead
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_SPCI: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        """
        self.alpha = alpha
        n1 = self.X_train.shape[0]
        self.past_window = (
            past_window  # For SPCI, this is the "lag" for predicting quantile
        )
        if smallT:
            # Namely, for special use of EnbPI, only use at most past_window number of LOO residuals.
            n1 = min(self.past_window, self.X_train.shape[0])
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma
        start = time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        if use_SPCI:
            s = stride
            stride = 1
        # NOTE, NOT ALL rows are actually "observable" in multi-step context, as this is rolling
        resid_strided = strided_app(
            self.Ensemble_online_resid[len(self.X_train) - n1 : -1], n1, stride
        )
        print(f"Shape of slided residual lists is {resid_strided.shape}")
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        # # NEW, alpha becomes alpha_t. Uncomment things below if we decide to use this upgraded EnbPI
        # alpha_t = alpha
        # errs = []
        # gamma = 0.005
        # method = 'simple'  # 'simple' or 'complex'
        # self.alphas = []
        # NOTE: 'max_features='log2', max_depth=2' make the model "simpler", which improves performance in practice
        self.QRF_ls = []
        self.i_star_ls = []
        for i in range(num_unique_resid):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    # Update QRF
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                # Use the fitted regressor.
                # NOTE, residX is NOT the same as before, as it depends on
                # "past_resid", which has most entries replaced.
                rfqr = self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all) / 2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid + i_star]
                width_left[i] = curr_SigmaX * wid_left
                width_right[i] = curr_SigmaX * wid_right
                num_print = int(num_unique_resid / 20)
                if num_print == 0:
                    print(
                        f"Width at test {i} is {width_right[i]-width_left[i]}"
                    )
                else:
                    if i % num_print == 0:
                        print(
                            f"Width at test {i} is {width_right[i]-width_left[i]}"
                        )
            else:  # use EnbPI
                past_resid = resid_strided[i, :]
                # Naive empirical quantile, where we use the SAME residuals for multi-step prediction
                # The number of bins will be determined INSIDE binning
                beta_hat_bin = binning(past_resid, alpha)
                # beta_hat_bin = binning(past_resid, alpha_t)
                self.beta_hat_bins.append(beta_hat_bin)
                # width_left[i] = curr_SigmaX * np.percentile(
                #     past_resid, math.ceil(100 * beta_hat_bin)
                # )
                # width_right[i] = curr_SigmaX * np.percentile(
                #     past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin))
                # )
        print(
            f"Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time()-start} secs."
        )
        # Ntest = len(out_sample_predict)
        # This is because |width|=T1/stride.
        # width_left = np.repeat(width_left, stride)[:Ntest]
        # This is because |width|=T1/stride.
        # width_right = np.repeat(width_right, stride)[:Ntest]
        PIs_Ensemble = pd.DataFrame(
            np.c_[
                out_sample_predict + width_left,
                out_sample_predict + width_right,
            ],
            columns=["lower", "upper"],
        )
        self.PIs_Ensemble = PIs_Ensemble
