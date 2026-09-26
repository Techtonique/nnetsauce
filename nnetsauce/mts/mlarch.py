import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt


def _is_full_mts(model):
    """True if `model` is a full ns.MTS instance (native fit(series)/
    predict(h=...) convention), False if it's a plain sklearn-style
    regressor (fit(X, y)/predict(X) convention). Checked via `.obj` --
    ns.MTS's constructor argument name, stored verbatim -- rather than
    `.lags` alone, since some plain regressors could coincidentally have a
    same-named hyperparameter."""
    return hasattr(model, "obj") and hasattr(model, "lags")


class MLARCH:
    """Machine Learning-agnostic ARCH for nearly-stationary time series (e.g., returns)

    Each of model_mean, model_sigma, model_residuals can independently be
    EITHER:
      - a full ns.MTS instance, e.g. ns.MTS(RidgeCV(), lags=5,
        type_pi="scp2-kde", replications=200) -- fit as its own univariate
        forecaster (fit(series), predict(h=...)), giving it access to
        ns.MTS's conformal/simulation machinery for that component; or
      - a plain sklearn regressor, e.g. RidgeCV() -- fit via a manually
        built lag matrix (fit(X, y), predict(X)), forecast via a hand-rolled
        recursive multi-step loop. Simpler, point-forecast only for that
        component (no .sims contribution from it).
    The two modes are auto-detected per component, so you can mix them
    (e.g. model_mean/model_residuals as ns.MTS, model_sigma as a bare
    regressor) exactly as needed.

    Parameters
    ----------
    model_mean : ns.MTS or sklearn regressor
        Model for mean component.
    model_sigma : ns.MTS or sklearn regressor
        Model for volatility component. If a plain regressor, its own lag
        order is controlled by `lags_vol` below (mirrors the original
        pre-fix behavior). If ns.MTS, its own `lags` constructor argument
        controls the ARCH order instead and `lags_vol` is ignored for it.
    model_residuals : ns.MTS or sklearn regressor
        Model for standardized residuals.
    lags_vol : int, default=10
        Lag order used ONLY for components that are plain sklearn
        regressors (not ns.MTS). Also used as a minimum-length sanity
        check regardless of mode.
    z_method : {"oof", "in_sample"}, default="oof"
        How the mean residuals and the volatility used to standardize them
        are obtained.
          - "in_sample" (behavior before this option existed): residuals and
            sigma are the components' in-sample fitted values. A flexible
            regressor (ExtraTrees, KNN, deep trees...) nearly interpolates
            log(eps^2) in-sample, so |z| ~ 1 for every point, z_std_ shrinks
            and the z distribution loses its tails -> predictive intervals
            too narrow out of sample.
          - "oof": expanding-window out-of-fold, one-step-ahead predictions
            (n_folds folds; the first block is used for training only).
            The standardized residuals then reflect genuine out-of-sample
            forecast errors, which is what predict() extrapolates. For ns.MTS
            components, the out-of-fold fits use a clone of `model.obj` on
            the raw lag matrix (same approximation as for the scp* types).
    n_folds : int, default=5
        Number of out-of-fold blocks for z_method="oof".
    floor_quantile : float or None, default=0.01
        Squared residuals are floored at this quantile of the strictly
        positive squared residuals before taking logs (scale-aware; avoids
        log(1e-8) ~ -18.4 outliers from zero returns). None restores the
        former absolute floor of 1e-8.
    """

    def __init__(self, model_mean, model_sigma, model_residuals, lags_vol=10,
                 z_method="oof", n_folds=5, floor_quantile=0.01):
        assert z_method in ("oof", "in_sample"), "z_method must be 'oof' or 'in_sample'"
        self.model_mean = model_mean
        self.model_sigma = model_sigma
        self.model_residuals = model_residuals
        self.lags_vol = lags_vol
        self.z_method = z_method
        self.n_folds = n_folds
        self.floor_quantile = floor_quantile

    def _create_lags(self, y, lags):
        """Create a lagged feature matrix. Used both for the plain-
        regressor calling convention, and for the auxiliary in-sample fit
        that ns.MTS components need (see fit() below)."""
        n = len(y)
        if n <= lags:
            raise ValueError(f"Series length {n} must be > lags {lags}")
        X = np.zeros((n - lags, lags))
        for i in range(lags):
            X[:, i] = y[i: (n - lags + i)]
        return X

    @staticmethod
    def _point_forecast(pred_result):
        """ns.MTS.predict(h=...) returns a plain DataFrame when type_pi is
        unset, or a DescribeResult namedtuple (with a .mean DataFrame) when
        type_pi is set. NOTE: can't discriminate via hasattr(x, "mean") --
        pd.DataFrame also has a .mean() *method* -- so check the DataFrame
        case explicitly first."""
        if isinstance(pred_result, pd.DataFrame):
            return pred_result.values.ravel()
        return pred_result.mean.values.ravel()

    def _fit_component_series(self, model, series, lags_fallback):
        """Fit `model` on `series` (a 1-D numpy array), whichever mode it
        is. Returns (fitted_in_sample_values, effective_lags) so callers
        can align other arrays against the resulting (shorter) length.

        For ns.MTS: fits natively on the series. In-sample fitted values
        come from model.residuals_ directly ("fitted = actual - residual")
        when that's reliable -- i.e. whenever type_pi does NOT start with
        "scp" or "scp2". For split-conformal type_pi variants, ns.MTS.fit()
        internally splits the series in half, refits on only one half, and
        residuals_ ends up half-length and index-misaligned with the
        original series -- silently wrong, not just short. In that case we
        fall back to a fresh CLONE of model.obj fit directly on the same
        raw lag matrix, purely to get a robust, correctly-aligned in-sample
        estimate. NOTE: this fallback approximates ns.MTS's own fit -- it
        does NOT replicate ns.MTS's internal quasi-randomized feature
        expansion (n_hidden_features, nodes_sim, etc.), only the raw lags,
        so it's a slightly simpler model than what ns.MTS actually fits.
        Acceptable for standardizing residuals; worth knowing about.

        For a plain regressor: fits directly on a manually built lag
        matrix (the original pre-fix calling convention), lag order given
        by lags_fallback (i.e. lags_vol).
        """
        if _is_full_mts(model):
            model.fit(series.reshape(-1, 1))
            lags = model.lags  # int even if lags was 'AIC'/'BIC' etc.
            type_pi = getattr(model, "type_pi", "")
            if isinstance(type_pi, str) and type_pi.startswith("scp"):
                X = self._create_lags(series, lags)
                y_ = series[lags:]
                aux = clone(model.obj)
                aux.fit(X, y_)
                fitted = aux.predict(X)
            else:
                resid = model.residuals_.ravel()
                fitted = series[lags:] - resid
            return fitted, lags
        else:
            lags = lags_fallback
            X = self._create_lags(series, lags)
            y_ = series[lags:]
            model.fit(X, y_)
            fitted = model.predict(X)
            return fitted, lags

    def _oof_component_series(self, model, series, lags_fallback):
        """Expanding-window out-of-fold one-step-ahead fitted values.

        Returns (oof_fitted, lags, n_skip): oof_fitted[i] predicts
        series[lags + n_skip + i] from a model trained only on rows before
        its block. The first n_skip rows (first block) have no OOF value.
        """
        lags = model.lags if _is_full_mts(model) else lags_fallback
        if isinstance(lags, str):  # 'AIC'/'BIC' etc. -> resolve via a full fit
            model.fit(series.reshape(-1, 1))
            lags = model.lags
        base = model.obj if _is_full_mts(model) else model
        X = self._create_lags(series, lags)
        y_ = series[lags:]
        n_rows = len(y_)
        bounds = np.linspace(0, n_rows, self.n_folds + 2).astype(int)
        n_skip = bounds[1]
        oof = np.empty(n_rows - n_skip)
        for k in range(1, self.n_folds + 1):
            start, stop = bounds[k], bounds[k + 1]
            m = clone(base)
            m.fit(X[:start], y_[:start])
            oof[start - n_skip: stop - n_skip] = m.predict(X[start:stop])
        return oof, lags, n_skip

    def _log_squared(self, resid):
        resid_squared = resid ** 2
        if self.floor_quantile is None:  # exact pre-patch behavior
            self.floor_ = 1e-8
            return np.log(resid_squared + 1e-8)
        else:
            pos = resid_squared[resid_squared > 0]
            self.floor_ = np.quantile(pos, self.floor_quantile) if pos.size else 1e-8
        return np.log(np.maximum(resid_squared, self.floor_))

    def _predict_component(self, model, h, last_lags=None):
        """Forecast `model` h steps ahead, whichever mode it is.

        For ns.MTS: delegates to its native multi-step recursive forecast.
        For a plain regressor: hand-rolled recursive loop (predict one
        step, append the prediction to the lag window, repeat) -- this is
        the same logic ns.MTS runs internally for its own components, just
        made explicit here since a bare regressor has no such machinery.
        `last_lags` (most recent `lags_vol` values of the fitted series) is
        required in this branch.
        """
        if _is_full_mts(model):
            return self._point_forecast(model.predict(h=h))
        else:
            if last_lags is None:
                raise ValueError("last_lags required for a plain-regressor component")
            forecast = np.zeros(h)
            current = last_lags.copy()
            for i in range(h):
                x_t = current.reshape(1, -1)
                forecast[i] = model.predict(x_t)[0]
                current = np.append(current[1:], forecast[i])
            return forecast

    def fit(self, y, **kwargs):
        """Fit the MLARCH model

        Parameters
        ----------
        y : array-like
            Target time series (should be stationary, e.g., returns)

        Returns
        -------
        self
        """
        # Format input
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        y = y.ravel()

        if len(y) < self.lags_vol + 20:
            raise ValueError(f"Need at least {self.lags_vol + 20} observations")

        if self.z_method == "in_sample":
            # Step 1: mean model -- forecasts y as its own series (ns.MTS mode)
            # or via a manual AR(lags_vol) lag regression (plain-regressor mode)
            fitted_mean, mean_lags = self._fit_component_series(self.model_mean, y, self.lags_vol)
            mean_residuals = y[mean_lags:] - fitted_mean
            self._mean_lags = mean_lags

            # Step 2: ARCH volatility model on log(squared residuals)
            log_resid_squared = self._log_squared(mean_residuals)
            fitted_log_sigma, sigma_lags = self._fit_component_series(
                self.model_sigma, log_resid_squared, self.lags_vol
            )
            self._sigma_lags = sigma_lags
            fitted_sigma = np.exp(fitted_log_sigma)
            eps_for_z = mean_residuals[sigma_lags:]
        else:
            # Step 1: mean model. Out-of-fold residuals give honest error
            # scale; the full-data fit is kept for forecasting.
            oof_mean, mean_lags, n_skip_m = self._oof_component_series(
                self.model_mean, y, self.lags_vol)
            mean_residuals = y[mean_lags + n_skip_m:] - oof_mean
            _, mean_lags = self._fit_component_series(self.model_mean, y, self.lags_vol)
            self._mean_lags = mean_lags

            # Step 2: volatility model on log(squared OOF residuals); OOF
            # one-step-ahead log-variance predictions standardize the residuals
            log_resid_squared = self._log_squared(mean_residuals)
            oof_log_sigma, sigma_lags, n_skip_s = self._oof_component_series(
                self.model_sigma, log_resid_squared, self.lags_vol)
            fitted_sigma = np.exp(oof_log_sigma)
            eps_for_z = mean_residuals[sigma_lags + n_skip_s:]
            _, sigma_lags = self._fit_component_series(
                self.model_sigma, log_resid_squared, self.lags_vol)
            self._sigma_lags = sigma_lags

        # Step 3: standardized residuals, aligned to fitted_sigma's length
        standardized_residuals = eps_for_z / np.sqrt(fitted_sigma)
        self.z_mean_ = np.mean(standardized_residuals)
        self.z_std_ = np.std(standardized_residuals)
        standardized_residuals = (
            standardized_residuals - self.z_mean_
        ) / self.z_std_
        self.standardized_residuals_ = standardized_residuals

        # Step 4: residuals model
        _, resid_lags = self._fit_component_series(
            self.model_residuals, standardized_residuals, self.lags_vol
        )
        self._resid_lags = resid_lags

        # Store state needed for plain-regressor recursive forecasting
        self._last_log_sigma_lags = log_resid_squared[-sigma_lags:].copy() \
            if not _is_full_mts(self.model_sigma) else None
        self._last_z_lags = standardized_residuals[-resid_lags:].copy() \
            if not _is_full_mts(self.model_residuals) else None
        self._last_y_lags = y[-mean_lags:].copy() \
            if not _is_full_mts(self.model_mean) else None

        # Store diagnostics
        self.fitted_volatility_mean_ = np.mean(np.sqrt(fitted_sigma))
        self.fitted_volatility_std_ = np.std(np.sqrt(fitted_sigma))

        return self

    def predict(self, h=5, level=95, return_sims=False):
        """Predict future values

        Parameters
        ----------
        h : int
            Forecast horizon
        level : int
            Confidence level for prediction intervals
        return_sims : bool
            If True, return full simulation paths (only available if
            model_residuals is a full ns.MTS with type_pi/replications set;
            falls back to Gaussian intervals otherwise, same as before)

        Returns
        -------
        DescribeResult
            Named tuple with mean, sims, lower, upper
        """
        DescribeResult = namedtuple(
            "DescribeResult", ("mean", "sims", "lower", "upper")
        )

        mean_forecast = self._predict_component(self.model_mean, h, self._last_y_lags)
        log_sigma_forecast = self._predict_component(self.model_sigma, h, self._last_log_sigma_lags)
        sigma_forecast = np.exp(log_sigma_forecast)
        z_forecast_normalized = self._predict_component(self.model_residuals, h, self._last_z_lags)
        z_forecast = z_forecast_normalized * self.z_std_ + self.z_mean_

        # Combine: μ + z × σ
        point_forecast = mean_forecast + z_forecast * np.sqrt(sigma_forecast)

        # Generate prediction intervals
        sims = None
        if return_sims and _is_full_mts(self.model_residuals):
            preds_z_for_sims = self.model_residuals.predict(h=h)
            # NOTE: current nnetsauce returns .sims as a tuple of B
            # per-replicate DataFrames (each shape (h, 1)), not a single
            # wide DataFrame with one column per replicate.
            if getattr(preds_z_for_sims, "sims", None) is not None:
                sims_z_normalized = preds_z_for_sims.sims
                n_sims = len(sims_z_normalized)

                sims = np.zeros((h, n_sims))
                for sim_idx in range(n_sims):
                    z_sim = (
                        sims_z_normalized[sim_idx].values.ravel() * self.z_std_
                        + self.z_mean_
                    )
                    sims[:, sim_idx] = mean_forecast + z_sim * np.sqrt(
                        sigma_forecast
                    )

                alpha = 1 - level / 100
                lower_bound = np.quantile(sims, alpha / 2, axis=1)
                upper_bound = np.quantile(sims, 1 - alpha / 2, axis=1)
            else:
                z_score = norm.ppf(1 - (1 - level / 100) / 2)
                margin = z_score * np.sqrt(sigma_forecast) * self.z_std_
                lower_bound = point_forecast - margin
                upper_bound = point_forecast + margin
        else:
            if return_sims and not _is_full_mts(self.model_residuals):
                import warnings
                warnings.warn(
                    "return_sims=True but model_residuals is a plain "
                    "sklearn regressor (no .sims available) -- falling "
                    "back to Gaussian intervals. Use ns.MTS with "
                    "type_pi=/replications= for model_residuals to get "
                    "genuine simulation-based intervals."
                )
            z_score = norm.ppf(1 - (1 - level / 100) / 2)
            margin = z_score * np.sqrt(sigma_forecast) * self.z_std_
            lower_bound = point_forecast - margin
            upper_bound = point_forecast + margin

        return DescribeResult(point_forecast, sims, lower_bound, upper_bound)