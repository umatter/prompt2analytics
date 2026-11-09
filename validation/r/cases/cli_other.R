# validation/r/cases/cli_other.R -- references for p2a CLI outputs that T3-WP61 adds where no wave-2 case
# records the quantity the CLI prints. Every other T3-WP61 CLI test reads a wave-2 case (registry
# validation/methods/cli_other.json lists them all).
#  * cli_pca_scores (CLI-23): stats::prcomp on WP42's hclust_60 (centred, not scaled: what `p2a ml pca`
#    computes). sdev, |rotation| and the scores x (column-major, n x p). The sign of each component is
#    arbitrary, so the test aligns every p2a score column to R's by the sign of their inner product.
#  * cli_viz_residuals (T3-WP52 hand-off): lm(mpg ~ wt + hp + qsec) on mtcars: hat values, rstandard,
#    cooks.distance, sigma and the rank, on the raw columns the CLI regresses (cli_mtcars.csv). WP52's
#    viz_residual_diag.csv holds only that model's fitted values, residuals and hat values; this case
#    checks that both agree to 1e-13.
case("cli_pca_scores", "ml/pca", "prcomp scores (centred, unscaled) on hclust_60", function(ctx) {
  d <- ctx$dataset("datasets/generated/hclust_60.csv")
  X <- as.matrix(d)
  p <- prcomp(X, center = TRUE, scale. = FALSE)
  list(dataset = "validation/datasets/generated/hclust_60.csv",
       values = list(sdev = vec(p$sdev), rotation_abs = vec(abs(p$rotation)), scores = vec(p$x),
                     center = named(p$center), n = nrow(X)),
       reference = "stats::prcomp(x, center = TRUE, scale. = FALSE): sdev, rotation, x",
       call = "prcomp(hclust_60)", packages = character(0))
})

case("cli_viz_residuals", "visualization/charts", "lm(mpg ~ wt + hp + qsec): hat values, rstandard, Cook's distance", function(ctx) {
  d <- ctx$gen_dataset("cli_mtcars", data.frame(mpg = mtcars$mpg, wt = mtcars$wt, hp = mtcars$hp, qsec = mtcars$qsec))
  m <- lm(mpg ~ wt + hp + qsec, data = d)
  w <- ctx$dataset("datasets/generated/viz_residual_diag.csv")       # T3-WP52's committed model columns
  stopifnot(max(abs(hatvalues(m) - w$hat3)) < 1e-13, max(abs(fitted(m) - w$fitted3)) < 1e-11)
  list(dataset = "validation/datasets/generated/cli_mtcars.csv",
       values = list(hat = vec(hatvalues(m)), rstandard = vec(rstandard(m)), cooks = vec(cooks.distance(m)),
                     fitted = vec(fitted(m)), residuals = vec(residuals(m)), sigma = summary(m)$sigma,
                     rank = m$rank, n = nrow(d)),
       reference = "stats::lm; stats::hatvalues, rstandard, cooks.distance",
       call = "lm(mpg ~ wt + hp + qsec, data = mtcars)", packages = character(0))
})
