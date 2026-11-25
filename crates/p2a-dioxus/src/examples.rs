//! Example datasets bundled with the app, offered on the welcome screen.
//!
//! Each example is a small CSV compiled into the binary and loaded into the
//! session through the `upload_dataset` tool, the same path a file chosen in
//! the browser takes, so the backend needs nothing extra.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dioxus::prelude::*;

use crate::api::ApiClient;
use crate::state::{SessionState, ensure_session};

/// One bundled example: what the welcome screen shows and what it loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Example {
    /// Dataset name in the session (also the file stem).
    pub name: &'static str,
    /// One-line description of the data.
    pub title: &'static str,
    /// The question the starter asks, shown under the title.
    pub question: &'static str,
    /// The prompt sent once the data is loaded (when a provider is set up).
    pub prompt: &'static str,
    /// CSV content.
    pub csv: &'static str,
    /// Rows and columns, for the dataset list.
    pub rows: usize,
    pub cols: usize,
}

/// The examples, in the order the welcome screen lists them.
pub const EXAMPLES: [Example; 3] = [
    Example {
        name: "grunfeld",
        title: "Investment by 10 US firms, 1935 to 1954",
        question: "Estimate a fixed effects model of investment on firm value and capital.",
        prompt: "Estimate a fixed effects panel model of inv on value and capital in grunfeld, with firm as the entity and year as the time variable.",
        csv: include_str!("../assets/examples/grunfeld.csv"),
        rows: 200,
        cols: 5,
    },
    Example {
        name: "lalonde",
        title: "Job training and earnings (NSW, LaLonde 1986)",
        question: "Did the training programme raise earnings in 1978?",
        prompt: "In lalonde, estimate the effect of treat on re78 with propensity score matching on age, educ, race, married, nodegree, re74 and re75.",
        csv: include_str!("../assets/examples/lalonde.csv"),
        rows: 614,
        cols: 9,
    },
    Example {
        name: "air_passengers",
        title: "Monthly airline passengers, 1949 to 1960",
        question: "Forecast the next 12 months with Holt-Winters.",
        prompt: "Fit a Holt-Winters model with multiplicative seasonality to passengers in air_passengers (monthly data, period 12), forecast the next 12 months and plot the forecast.",
        csv: include_str!("../assets/examples/air_passengers.csv"),
        rows: 144,
        cols: 2,
    },
];

/// Loads an example into the tab's session and adds it to the dataset list.
pub async fn load_example(
    mut session: Signal<SessionState>,
    example: Example,
) -> Result<(), String> {
    let session_id = ensure_session(session).await?;
    let args = serde_json::json!({
        "content": STANDARD.encode(example.csv.as_bytes()),
        "filename": format!("{}.csv", example.name),
        "name": example.name,
        "file_type": "csv",
    });
    let result = ApiClient::new()
        .call_tool(&session_id, "upload_dataset", args)
        .await?;
    if !result.success {
        return Err(result
            .error
            .unwrap_or_else(|| format!("Could not load {}", example.name)));
    }
    session
        .write()
        .add_dataset(example.name.to_string(), example.rows, example.cols);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_example_has_the_rows_and_columns_it_claims() {
        for ex in EXAMPLES {
            let mut lines = ex.csv.lines().filter(|l| !l.trim().is_empty());
            let header = lines.next().expect("header");
            assert_eq!(header.split(',').count(), ex.cols, "{} columns", ex.name);
            assert_eq!(lines.count(), ex.rows, "{} rows", ex.name);
        }
    }

    #[test]
    fn each_prompt_names_its_dataset() {
        for ex in EXAMPLES {
            assert!(ex.prompt.contains(ex.name), "{}", ex.name);
        }
    }
}
