# validation/r/cases/econ_panel_na.R -- panel estimators on data with missing cells, missing
# identifiers and string or numeric entity ids (T2-WP08: ECON-31, ECON-37, ECON-71). plm and
# fixest drop incomplete rows (na.omit); the reference is the estimate on the complete rows of the
# columns each estimator uses: FE, RE and the mean-group estimator take no time column, the
# two-way HDFE uses the year as well.
suppressPackageStartupMessages({ library(plm); library(fixest) })

case("panel_grunfeld_na", "econometrics/panel", "FE, RE, two-way HDFE and mean group on Grunfeld with missing cells and missing string/numeric firm ids", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  g$firm_name <- sprintf("F%02d", g$firm)                       # string entity id
  g$firm_id <- g$firm - 1L                                      # numeric entity id 0..9: a real id 0 exists (ECON-37)
  g$inv[c(5, 50)] <- NA; g$value[120] <- NA
  g$firm_name[c(77, 133)] <- NA; g$firm_id[c(77, 133)] <- NA    # two missing ids in different firms
  g$year[c(31, 150)] <- NA                                      # two missing periods in different firms and years
  d <- ctx$gen_dataset("grunfeld_na", g[, c("firm", "firm_id", "firm_name", "year", "inv", "value", "capital")])
  stopifnot(is.integer(d$firm_id), is.integer(d$year), is.character(d$firm_name),
            identical(is.na(d$firm_name), is.na(d$firm_id)))
  ce <- d[complete.cases(d[, c("inv", "value", "capital", "firm_name")]), ]   # rows FE / RE / pmg use
  cc <- d[complete.cases(d), ]                                                # rows the two-way HDFE uses
  stopifnot(nrow(d) - nrow(ce) == 5, nrow(d) - nrow(cc) == 7)
  pe <- pdata.frame(ce, index = "firm_name")   # one index: plm numbers periods within firm; FE/RE do not use them
  fe <- plm(inv ~ value + capital, data = pe, model = "within")
  re <- plm(inv ~ value + capital, data = pe, model = "random")
  hd <- feols(inv ~ value + capital | firm_name + year, data = d, vcov = "iid")
  mg <- pmg(inv ~ value + capital, data = pdata.frame(ce, index = "firm_id"), model = "mg")
  list(values = list(
    fe_coef = named(coef(fe)), fe_se = named(sqrt(diag(vcov(fe)))), fe_nobs = nobs(fe),
    re_coef = named(coef(re)), re_se = named(sqrt(diag(vcov(re)))), re_nobs = nobs(re),
    n_complete_fe = nrow(ce), n_groups = length(unique(ce$firm_name)),
    hdfe_coef = named(coef(hd)), hdfe_se_iid = named(se(hd)), hdfe_nobs = nobs(hd),
    hdfe_n_removed = nrow(d) - nobs(hd), n_complete = nrow(cc), n_years = length(unique(cc$year)),
    mg_coef = named(coef(mg)), mg_se = named(sqrt(diag(vcov(mg)))),
    missing_inv = sum(is.na(d$inv)), missing_value = sum(is.na(d$value)),
    missing_firm_name = sum(is.na(d$firm_name)), missing_year = sum(is.na(d$year))),
    reference = "plm::plm(model = 'within' | 'random') and plm::pmg(model = 'mg') on the complete rows of (inv, value, capital, firm id); fixest::feols(| firm_name + year) on the complete rows of all columns (both drop rows with NA in y, x or the index)",
    call = "plm(inv ~ value + capital, data = pdata.frame(ce, index = 'firm_name'), model = 'within' | 'random'); feols(inv ~ value + capital | firm_name + year, d, vcov = 'iid'); pmg(inv ~ value + capital, pdata.frame(ce, index = 'firm_id'), model = 'mg')",
    packages = c("plm", "fixest"))
})
