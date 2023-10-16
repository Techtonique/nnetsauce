#' Plot multivariate time series forecast or residuals
#'
#' @param x result from \code{basicf}, \code{ridge2f} or \code{varf} (multivariate time series forecast)
#' @param selected_series name of the time series selected for plotting
#' @param type "pi": basic prediction intervals;
#' "dist": a distribution of predictions; "sims": the simulations
#' @param level confidence levels for prediction intervals
#' @param ... additional parameters to be passed to \code{plot} or \code{matplot}
#'
#' @export
#'
#' @examples
#'
#' require(fpp)
#'
#' fit_obj_VAR <- ahead::varf(fpp::insurance, lags = 2,
#' h = 10, level = 95)
#'
#' fit_obj_ridge2 <- ahead::ridge2f(fpp::insurance, lags = 2,
#' h = 10, level = 95)
#'
#' par(mfrow=c(2, 2))
#' plot(fit_obj_VAR, "Quotes")
#' plot(fit_obj_VAR, "TV.advert")
#' plot(fit_obj_ridge2, "Quotes")
#' plot(fit_obj_ridge2, "TV.advert")
#'
#' obj <- ahead::ridge2f(fpp::insurance, h = 10, type_pi = "blockbootstrap",
#' block_length=5, B = 10)
#' par(mfrow=c(1, 2))
#' plot(obj, selected_series = "Quotes", type = "sims",
#' main = "Predictive simulation for Quotes")
#' plot(obj, selected_series = "TV.advert", type = "sims",
#' main = "Predictive simulation for TV.advert")
#'
#'
#' par(mfrow=c(1, 2))
#' plot(obj, selected_series = "Quotes", type = "dist",
#' main = "Predictive simulation for Quotes")
#' plot(obj, selected_series = "TV.advert", type = "dist",
#' main = "Predictive simulation for TV.advert")
#'
#'
plot.MTS <- function(x, selected_series,
                     level = 95, ...)
{
  if (!is.null(x$start) && !is.null(x$frequency))
  {
    y <- ts(x$df_[, selected_series],
            start = x$start,
            frequency = x$frequency)
  } else {
    warning("object has no attributes 'start' and 'frequency'")
    y <- ts(x$df_[, selected_series])
  }

  mean_fcast <- x$mean_[, selected_series]
  upper_fcast <- x$upper_[, selected_series]
  lower_fcast <- x$lower_[, selected_series]

  start_y <- x$start
  frequency_y <- x$frequency

  y_mean <- ts(c(y, mean_fcast), start = start_y,
               frequency = frequency_y)
  y_upper <- ts(c(y, upper_fcast), start = start_y,
                frequency = frequency_y)
  y_lower <- ts(c(y, lower_fcast), start = start_y,
                frequency = frequency_y)

  plot(y_mean, type='l',
       main=paste0("Forecasts for ", selected_series, " (", x$method, ")"),
       ylab="", ylim = c(min(c(y_upper, y_lower)),
                         max(c(y_upper, y_lower))), ...)
  lines(y_upper, col="gray60")
  lines(y_lower, col="gray60")
  lines(y_mean)
}
