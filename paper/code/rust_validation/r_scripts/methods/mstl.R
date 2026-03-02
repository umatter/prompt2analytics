# mstl.R - Multiple seasonal decomposition using forecast::mstl

suppressPackageStartupMessages({
  library(forecast)
})

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       frequency = 12, seasonal_periods = NULL) {

  # Extract time series
  y <- data[[dep_var]]

  # Create msts object if multiple seasonal periods given
  if (!is.null(seasonal_periods)) {
    ts_data <- msts(y, seasonal.periods = seasonal_periods)
  } else {
    ts_data <- ts(y, frequency = frequency)
  }

  # Fit MSTL decomposition
  decomp <- mstl(ts_data)

  # Extract components
  components <- as.data.frame(decomp)

  list(
    trend = as.list(components$Trend),
    remainder = as.list(components$Remainder),
    seasonal_columns = names(components)[grepl("^Seasonal", names(components))],
    n_obs = length(y),
    frequency = frequency
  )
}
