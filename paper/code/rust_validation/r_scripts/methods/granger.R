# granger.R - Granger causality test using lmtest::grangertest

suppressPackageStartupMessages({
  library(lmtest)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       order = 1) {

  # dep_var is the dependent variable (y)
  # indep_vars[1] is the potential cause (x)
  y <- data[[dep_var]]
  x <- data[[indep_vars[1]]]

  # Test whether x Granger-causes y
  df <- data.frame(y = y, x = x)
  result <- grangertest(y ~ x, order = order, data = df)

  # Extract F statistic and p-value (row 2 contains the test)
  f_stat <- result$F[2]
  p_value <- result$`Pr(>F)`[2]
  df1 <- result$Df[2]
  df2 <- result$Res.Df[2]

  # Also test reverse direction: y -> x
  result_reverse <- grangertest(x ~ y, order = order, data = df)
  f_stat_reverse <- result_reverse$F[2]
  p_value_reverse <- result_reverse$`Pr(>F)`[2]

  list(
    f_statistic = f_stat,
    p_value = p_value,
    df1 = abs(df1),
    df2 = df2,
    order = order,
    direction = paste(indep_vars[1], "->", dep_var),
    f_statistic_reverse = f_stat_reverse,
    p_value_reverse = p_value_reverse,
    direction_reverse = paste(dep_var, "->", indep_vars[1]),
    n_obs = length(y)
  )
}
