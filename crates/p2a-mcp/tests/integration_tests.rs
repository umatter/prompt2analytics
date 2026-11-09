//! Integration tests: every handler module reached through `call_tool_with_session` (T3-WP58, TOOL-31). Each
//! test sends JSON arguments through the real dispatcher (request parsing, limits, column extraction, output),
//! on R case datasets with integer and missing cells, and checks the answer against the R case
//! (validation/r/cases/*.R) or a property that needs none (stated in the test). Tools whose numbers belong to
//! other packages' Tier 4 work are smoke-tested.
use p2a_mcp::AnalyticsServer;

/// The router exposes tools from every handler module (stdio and HTTP builds alike).
#[test]
fn test_router_lists_every_handler_module() {
    let names: Vec<String> = AnalyticsServer::new()
        .router_tools()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    let mut expected = vec![
        "load_dataset",
        "set_seed",
        "viz_histogram",
        "ml_kmeans",
        "compute_correlation",
        "hypothesis_t_test",
        "regression_ols",
        "panel_fixed_effects",
        "logit",
        "iv_2sls",
        "timeseries_acf",
        "moran_test",
        "munge_filter",
        "cox_ph",
        "data_quality_profile",
        "search_tools",
    ];
    if cfg!(feature = "database") {
        expected.push("db_duckdb_query");
    }
    for e in expected {
        assert!(names.iter().any(|n| n == e), "{e} is not routed");
    }
}

#[cfg(feature = "http")]
#[path = "tier3/wp58_text.rs"]
mod checks;
#[cfg(feature = "http")]
#[path = "tier3/common.rs"]
mod common;

#[cfg(feature = "http")]
mod dispatch {
    use crate::checks::*;
    use crate::common::{call, session_with_datasets, text};
    use p2a_core::polars::prelude::DataType;
    use p2a_mcp::transport::http::ContentItem;
    use p2a_validation::{Case, Check, Tol};
    use serde_json::json;

    fn col_strings(ds: &p2a_core::data::Dataset, name: &str) -> Vec<Option<String>> {
        ds.df()
            .column(name)
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }
    fn col_i64(ds: &p2a_core::data::Dataset, name: &str) -> Vec<Option<i64>> {
        ds.df()
            .column(name)
            .unwrap()
            .as_materialized_series()
            .i64()
            .unwrap()
            .into_iter()
            .collect()
    }

    /// data: create_dataset infers Int64 and nulls; list_datasets and head_dataset report them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dispatch_data_module() {
        let (session, server) = session_with_datasets(vec![]);
        let csv = "id,x,k\n1,1.5,3\n2,,4\n3,2.5,5\n4,3.5,6\n5,4.5,7\n";
        assert!(
            call(
                &server,
                &session,
                "create_dataset",
                json!({"name": "d", "csv_content": csv})
            )
            .await
            .success
        );
        let ds = session.datasets.read().await.get("d").cloned().expect("d");
        assert_eq!(ds.df().column("k").unwrap().dtype(), &DataType::Int64);
        assert_eq!(ds.df().column("x").unwrap().null_count(), 1);
        assert!(
            text(&call(&server, &session, "list_datasets", json!({})).await)
                .contains("**d**: 5 rows x 3 columns")
        );
        let h = text(
            &call(
                &server,
                &session,
                "head_dataset",
                json!({"dataset": "d", "n": 2}),
            )
            .await,
        );
        assert!(h.contains("1.5") && !h.contains("3.5"), "{h}");
    }

    /// utils: the session seed round-trips; seeded random data is reproducible.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dispatch_utils_module() {
        let (session, server) = session_with_datasets(vec![]);
        assert!(
            call(&server, &session, "set_seed", json!({"seed": 7}))
                .await
                .success
        );
        assert!(
            text(&call(&server, &session, "get_seed", json!({})).await)
                .contains("Current global seed: 7")
        );
        for name in ["r1", "r2"] {
            let a = json!({"n_rows": 5, "seed": 3, "name": name,
                           "columns": [{"name": "z", "distribution": {"type": "normal", "mean": 0.0, "std": 1.0}}]});
            assert!(
                call(&server, &session, "generate_random_data", a)
                    .await
                    .success
            );
        }
        let g = session.datasets.read().await;
        let z = |n: &str| {
            g.get(n)
                .unwrap()
                .df()
                .column("z")
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>()
        };
        assert_eq!(z("r1").len(), 5);
        assert_eq!(z("r1"), z("r2"));
    }

    /// viz: a histogram comes back as a PNG image (a property test: no R value).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dispatch_viz_module() {
        let (session, server) =
            session_with_datasets(vec![("mcp_dispatch", generated_dataset("mcp_dispatch"))]);
        let r = call(
            &server,
            &session,
            "viz_histogram",
            json!({"dataset": "mcp_dispatch", "column": "a", "bins": 10}),
        )
        .await;
        assert!(r.success, "{}", text(&r));
        let png = r
            .content
            .iter()
            .find_map(|c| match c {
                ContentItem::Image { data, mime_type } if mime_type == "image/png" => {
                    Some(data.clone())
                }
                _ => None,
            })
            .expect("a PNG image");
        assert!(png.starts_with("iVBORw0KGgo"), "base64 PNG signature");
    }

    /// regression: OLS on Grunfeld (integer firm and year), HC1 SEs by default (ols_grunfeld, T1).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_regression_module() {
        let case = Case::load("ols_grunfeld");
        let (session, server) = session_with_datasets(vec![("grunfeld", dataset_of(&case))]);
        let t = text(
            &call(
                &server,
                &session,
                "regression_ols",
                json!({"dataset": "grunfeld", "y": "inv", "x": ["value", "capital"]}),
            )
            .await,
        );
        for v in ["(Intercept)", "value", "capital"] {
            assert_row(
                v,
                row(&t, v),
                &[case.named("coef", v), case.named("se_hc1", v)],
                Tol::printed(4),
            );
        }
    }

    /// panel: random effects on Grunfeld as plm (panel_re_grunfeld, T1).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_panel_module() {
        let case = Case::load("panel_re_grunfeld");
        let (session, server) = session_with_datasets(vec![("grunfeld", dataset_of(&case))]);
        let t = text(&call(&server, &session, "panel_random_effects", json!({"dataset": "grunfeld", "y": "inv", "x": ["value", "capital"], "entity_var": "firm"})).await);
        for v in ["(Intercept)", "value", "capital"] {
            assert_row(
                v,
                row(&t, v),
                &[case.named("coef", v), case.named("se", v)],
                Tol::printed(4),
            );
        }
        assert_shows(
            "Wald",
            after(&t, "Wald chi-squared:"),
            case.scalar("wald_chisq"),
            Tol::printed(4),
        );
        assert_shows("theta", &t, case.scalar("theta_mean"), Tol::printed(4));
    }

    /// discrete: probit as glm (disc_binary_glm, T3-WP18; y and d are Int64).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_discrete_module() {
        let case = Case::load("disc_binary_glm");
        let (session, server) = session_with_datasets(vec![("binary_glm", dataset_of(&case))]);
        let t = text(
            &call(
                &server,
                &session,
                "probit",
                json!({"dataset": "binary_glm", "y": "y", "x": ["x1", "x2", "d"]}),
            )
            .await,
        );
        for v in ["(Intercept)", "x1", "x2", "d"] {
            assert_row(
                v,
                row(&t, v),
                &[
                    case.named("probit_coef_tight", v),
                    case.named("probit_se_tight", v),
                ],
                Tol::printed(4),
            );
        }
    }

    /// survival: log-rank by sex on lung (integer time, status 1/2) as survdiff (surv_logrank_lung, T2-WP09).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_survival_module() {
        let case = Case::load("surv_logrank_lung");
        let (session, server) = session_with_datasets(vec![("lung", dataset_of(&case))]);
        let t = text(
            &call(
                &server,
                &session,
                "log_rank",
                json!({"dataset": "lung", "time": "time", "event": "status", "group": "sex"}),
            )
            .await,
        );
        assert_shows(
            "chi2",
            after(&t, "Chi-squared ="),
            case.scalar("sex_chisq"),
            Tol::printed(4),
        );
    }

    /// causal: 2SLS with two endogenous regressors as AER::ivreg (iv_two_endog, T2-WP08); coefficients only (the
    /// SE type is WP55's surface).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_causal_module() {
        let case = Case::load("iv_two_endog");
        let (session, server) = session_with_datasets(vec![("iv_two_endog", dataset_of(&case))]);
        let t = text(
            &call(
                &server,
                &session,
                "iv_2sls",
                json!({"dataset": "iv_two_endog", "y": "y", "x_exog": ["w1", "w2"],
            "x_endog": ["x1", "x2"], "instruments": ["z1", "z2", "z3"]}),
            )
            .await,
        );
        for v in ["x1", "x2", "w1", "w2"] {
            assert_row(v, row(&t, v), &[case.named("coef", v)], Tol::printed(4));
        }
        assert_shows(
            "n",
            after(&t, "No. Observations:"),
            case.scalar("n_obs"),
            Tol::exact(),
        );
    }

    /// stats: two-way ANOVA where y and fa are missing in a row whose fb is present (the null-alignment fix of
    /// 5f86b49, through the dispatcher), MANOVA Pillai (T2-WP07), and factanal (smoke: STAT-23/STAT-49 are Tier 4).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_stats_module() {
        let (av, mv, md) = (
            Case::load("stat_anova_two_way_unbalanced"),
            Case::load("stat_manova_na"),
            Case::load("mcp_dispatch_smoke"),
        );
        let (session, server) = session_with_datasets(vec![
            ("anova_two_way_unbalanced", dataset_of(&av)),
            ("manova_na", dataset_of(&mv)),
            ("mcp_dispatch", dataset_of(&md)),
        ]);
        let t = text(&call(&server, &session, "anova_two_way", json!({"dataset": "anova_two_way_unbalanced", "response": "y", "factor_a": "fa", "factor_b": "fb"})).await);
        for k in [
            "int_fa_ss",
            "int_fb_ss",
            "int_fa_x_fb_ss",
            "int_fa_f",
            "int_fb_f",
            "int_fa_x_fb_f",
            "int_resid_ss",
        ] {
            assert_shows(k, &t, av.scalar(k), Tol::printed(4));
        }
        let t = text(&call(&server, &session, "anova_manova", json!({"dataset": "manova_na", "response_vars": ["y1", "y2"], "factor": "g", "test": "pillai"})).await);
        assert_shows("Pillai", &t, mv.scalar("pillai_stat"), Tol::printed(4));
        assert_shows(
            "approx F",
            &t,
            mv.scalar("pillai_approx_f"),
            Tol::printed(4),
        );
        let r = call(
            &server,
            &session,
            "multivariate_factanal",
            json!({"dataset": "mcp_dispatch", "columns": ["f1", "f2", "f3", "f4"], "n_factors": 1}),
        )
        .await;
        assert!(r.success && text(&r).contains("f1"), "{}", text(&r));
    }

    /// hypothesis: Fisher's exact test on 2000 rows, chi-square independence with small expected counts and the
    /// goodness-of-fit test with missing categories (T2-WP05).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_hypothesis_module() {
        let (fi, ci, cg) = (
            Case::load("stat_fisher_rows_2000"),
            Case::load("stat_chisq_small_expected"),
            Case::load("stat_chisq_gof"),
        );
        let (session, server) = session_with_datasets(vec![
            ("fisher_rows_2000", dataset_of(&fi)),
            ("chisq_small_expected", dataset_of(&ci)),
            ("chisq_gof", dataset_of(&cg)),
        ]);
        let t = text(
            &call(
                &server,
                &session,
                "hypothesis_fisher_exact",
                json!({"dataset": "fisher_rows_2000", "row_var": "treat", "col_var": "outcome"}),
            )
            .await,
        );
        assert_shows("Fisher p", &t, fi.scalar("p_two_sided"), Tol::printed(4));
        let t = text(
            &call(
                &server,
                &session,
                "hypothesis_chisq_independence",
                json!({"dataset": "chisq_small_expected", "row_var": "x", "col_var": "y"}),
            )
            .await,
        );
        assert_shows("chi2", &t, ci.scalar("chisq"), Tol::printed(4));
        assert_shows("p", &t, ci.scalar("p"), Tol::printed(4));
        let t = text(
            &call(
                &server,
                &session,
                "hypothesis_chisq_gof",
                json!({"dataset": "chisq_gof", "column": "cat"}),
            )
            .await,
        );
        assert_shows("GOF chi2", &t, cg.scalar("uniform_chisq"), Tol::printed(4));
    }

    /// timeseries: PACF of an Int64 series (cli_stat_acf_int), the lag-0 CCF (= cor) and a mean shift found by
    /// PELT at R's location; Holt-Winters and VAR IRF are smoke tests (pinned by T3-WP37 and WP12).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_timeseries_module() {
        let (ac, md) = (
            Case::load("cli_stat_acf_int"),
            Case::load("mcp_dispatch_smoke"),
        );
        let (session, server) = session_with_datasets(vec![
            ("friedman_quade", dataset_of(&ac)),
            ("mcp_dispatch", dataset_of(&md)),
        ]);
        let j = json_of(&call(&server, &session, "timeseries_acf", json!({"dataset": "friedman_quade", "column": "y", "lag_max": 5, "acf_type": "partial"})).await);
        Check::new(&ac)
            .vector("pacf", &nums(&j["values"]), Tol::coef(24))
            .finish();
        let j = json_of(
            &call(
                &server,
                &session,
                "timeseries_ccf",
                json!({"dataset": "mcp_dispatch", "x": "u", "y": "w", "lag_max": 3}),
            )
            .await,
        );
        let lags = nums(&j["lags"]);
        let at0 = lags.iter().position(|l| *l == 0.0).expect("lag 0");
        Check::new(&md)
            .scalar("ccf_lag0", nums(&j["values"])[at0], Tol::coef(48))
            .finish();
        let t = text(
            &call(
                &server,
                &session,
                "ts_changepoint",
                json!({"dataset": "mcp_dispatch", "column": "s", "change_type": "mean"}),
            )
            .await,
        );
        assert_row(
            "changepoint",
            after(&t, "Changepoint Positions:"),
            &md.vector("cpt_mean_location"),
            Tol::exact(),
        );
        let r = call(
            &server,
            &session,
            "ts_holt_winters",
            json!({"dataset": "mcp_dispatch", "column": "seas", "period": 4, "horizon": 4}),
        )
        .await;
        assert!(r.success && text(&r).contains("t+4"), "{}", text(&r));
        let r = call(
            &server,
            &session,
            "ts_var_irf",
            json!({"dataset": "mcp_dispatch", "columns": ["y1", "y2"], "lags": 1, "steps": 4}),
        )
        .await;
        assert!(r.success, "{}", text(&r));
    }

    /// ml: k-means on three separated blobs reaches R's optimum (kmeans(nstart = 50)).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_ml_module() {
        let case = Case::load("mcp_dispatch_smoke");
        let (session, server) = session_with_datasets(vec![("mcp_dispatch", dataset_of(&case))]);
        let t = text(&call(&server, &session, "ml_kmeans", json!({"dataset": "mcp_dispatch", "columns": ["a", "b"], "k": 3, "n_init": 10, "seed": 1})).await);
        assert_shows(
            "WCSS",
            after(&t, "WCSS"),
            case.scalar("kmeans_tot_withinss"),
            Tol::printed(4),
        );
        assert_eq!(t.matches(": 16 points").count(), 3, "{t}");
    }

    /// spatial: rook weights on an 8 x 6 grid stored in the session, then Moran's I (T2-WP02 keeps the weights).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_spatial_module() {
        let case = Case::load("mcp_dispatch_smoke");
        let (session, server) = session_with_datasets(vec![("mcp_dispatch", dataset_of(&case))]);
        let r = call(
            &server,
            &session,
            "spatial_neighbors",
            json!({"dataset": "mcp_dispatch", "x_coord": "gx", "y_coord": "gy",
            "method": "distance", "d_max": 1.1, "style": "W", "weights_name": "rook"}),
        )
        .await;
        assert!(r.success, "{}", text(&r));
        assert_shows(
            "links / n",
            after(&text(&r), "Average neighbors:"),
            case.scalar("n_rook_links") / case.scalar("n"),
            Tol::printed(2),
        );
        let t = text(
            &call(
                &server,
                &session,
                "moran_test",
                json!({"dataset": "mcp_dispatch", "variable": "v", "weights": "rook"}),
            )
            .await,
        );
        assert_shows(
            "I",
            after(&t, "Moran I statistic:"),
            case.scalar("moran_i"),
            Tol::printed(6),
        );
        assert_shows(
            "E[I]",
            after(&t, "Expected I:"),
            case.scalar("moran_expected"),
            Tol::printed(6),
        );
    }

    /// munging: regex replace / extract / count, split, concat and length on a text column (expected strings by
    /// construction: txt = fruit-<7 id>, "date" for every fourth row; counts and lengths from R).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_munging_module() {
        let case = Case::load("mcp_dispatch_smoke");
        let (session, server) = session_with_datasets(vec![("mcp_dispatch", dataset_of(&case))]);
        let steps = [
            (
                "str_regex_replace",
                json!({"dataset": "mcp_dispatch", "column": "txt", "pattern": "[0-9]+", "replacement": "#", "result_name": "m_rep"}),
            ),
            (
                "str_regex_extract",
                json!({"dataset": "mcp_dispatch", "column": "txt", "pattern": "-([0-9]+)$", "new_column": "num", "group": 1, "result_name": "m_ext"}),
            ),
            (
                "str_regex_count",
                json!({"dataset": "mcp_dispatch", "column": "txt", "pattern": "[aeiou]", "new_column": "vowels", "result_name": "m_cnt"}),
            ),
            (
                "str_split",
                json!({"dataset": "mcp_dispatch", "column": "txt", "pattern": "-", "prefix": "part", "result_name": "m_spl"}),
            ),
            (
                "str_concat",
                json!({"dataset": "mcp_dispatch", "columns": ["txt", "txt"], "new_column": "cc", "separator": "|", "result_name": "m_cat"}),
            ),
            (
                "str_length",
                json!({"dataset": "mcp_dispatch", "column": "txt", "new_column": "len", "result_name": "m_len"}),
            ),
        ];
        for (tool, args) in steps {
            let r = call(&server, &session, tool, args).await;
            assert!(r.success, "{tool}: {}", text(&r));
        }
        let g = session.datasets.read().await;
        let ds = |n: &str| g.get(n).cloned().unwrap_or_else(|| panic!("{n}"));
        assert_eq!(
            &col_strings(&ds("m_rep"), "txt")[..4],
            &[
                Some("apple-#".into()),
                Some("banana-#".into()),
                Some("cherry-#".into()),
                Some("date".into())
            ]
        );
        assert_eq!(
            &col_strings(&ds("m_ext"), "num")[..4],
            &[Some("7".into()), Some("14".into()), Some("21".into()), None]
        );
        let vowels: Vec<f64> = col_i64(&ds("m_cnt"), "vowels")
            .into_iter()
            .map(|v| v.unwrap() as f64)
            .collect();
        Check::new(&case)
            .vector("vowel_count", &vowels, Tol::exact())
            .finish();
        assert_eq!(
            &col_strings(&ds("m_spl"), "part_1")[..4],
            &[Some("7".into()), Some("14".into()), Some("21".into()), None]
        );
        assert_eq!(
            col_strings(&ds("m_cat"), "cc")[0],
            Some("apple-7|apple-7".into())
        );
        let len: Vec<f64> = col_i64(&ds("m_len"), "len")
            .into_iter()
            .map(|v| v.unwrap() as f64)
            .collect();
        Check::new(&case)
            .vector("txt_length", &len, Tol::exact())
            .finish();
    }

    /// cleaning: the quality profile counts the three missing xn cells.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_validate_dispatch_cleaning_module() {
        let case = Case::load("mcp_dispatch_smoke");
        let (session, server) = session_with_datasets(vec![("mcp_dispatch", dataset_of(&case))]);
        let t = text(
            &call(
                &server,
                &session,
                "data_quality_profile",
                json!({"dataset": "mcp_dispatch"}),
            )
            .await,
        );
        assert_row(
            "xn nulls",
            after(section(&t, "## xn"), "Null:"),
            &[case.scalar("xn_missing")],
            Tol::exact(),
        );
    }

    /// search: tool_info returns the generated registry entry with the input schema (WP05).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dispatch_search_module() {
        let (session, server) = session_with_datasets(vec![]);
        let t = text(
            &call(
                &server,
                &session,
                "tool_info",
                json!({"tool_name": "regression_glmnet"}),
            )
            .await,
        );
        assert!(
            t.contains("regression_glmnet") && t.contains("Input schema") && t.contains("lambda"),
            "{t}"
        );
    }

    /// database: one read-only statement on an in-memory DuckDB becomes a dataset. Feature `database`, which
    /// the CI feature set leaves out (it builds duckdb-sys), so this test runs locally only.
    #[cfg(feature = "database")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_dispatch_database_module() {
        let (session, server) = session_with_datasets(vec![]);
        let r = call(
            &server,
            &session,
            "db_duckdb_query",
            json!({"db_path": ":memory:", "query": "SELECT 2 + 3 AS v", "name": "q"}),
        )
        .await;
        assert!(r.success, "{}", text(&r));
        let ds = session.datasets.read().await.get("q").cloned().expect("q");
        assert_eq!(ds.df().height(), 1);
    }
}
