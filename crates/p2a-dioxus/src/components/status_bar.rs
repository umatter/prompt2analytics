//! The black status bar along the bottom of the window: backend state, the
//! model in use, how many datasets the session holds, the tool count and the
//! backend version.

use dioxus::prelude::*;

use crate::api::ApiClient;
use crate::state::{KeySource, Provider, SessionState, Settings};

/// What the backend reported at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Backend {
    Checking,
    Online {
        version: String,
        tools: Option<usize>,
    },
    Offline,
}

/// The model part of the bar: the model and whose key it uses.
fn model_label(settings: &Settings) -> String {
    match settings.provider {
        Provider::None => "no model connected".to_string(),
        Provider::Ollama => format!("{}, local", settings.current_model()),
        p => match settings.key_source(p) {
            KeySource::Environment => format!("{}, server key", settings.current_model()),
            _ => format!("{}, your key", settings.current_model()),
        },
    }
}

/// "1 dataset", "3 datasets".
fn count_label(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

/// The status bar.
#[component]
pub fn StatusBar() -> Element {
    let session = use_context::<Signal<SessionState>>();
    let settings = use_context::<Signal<Settings>>();
    let mut backend = use_signal(|| Backend::Checking);

    use_future(move || async move {
        let client = ApiClient::new();
        match client.health().await {
            Ok(h) => {
                let tools = client.list_tools().await.ok().map(|t| t.len());
                backend.set(Backend::Online {
                    version: h.version,
                    tools,
                });
            }
            Err(_) => backend.set(Backend::Offline),
        }
    });

    let datasets = session.read().loaded_datasets.len();
    let model = model_label(&settings.read());
    let state = backend.read().clone();

    rsx! {
        footer { class: "k-status", role: "status",
            match state {
                Backend::Checking => rsx! {
                    span { span { class: "k-orange k-spin" } " connecting" }
                },
                Backend::Online { .. } => rsx! {
                    span { span { class: "k-orange", "●" } " online" }
                },
                Backend::Offline => rsx! {
                    span { span { class: "k-orange", "○" } " backend not reachable" }
                },
            }
            span { "{model}" }
            span { {count_label(datasets, "dataset", "datasets")} }
            if let Backend::Online { tools: Some(n), .. } = &*backend.read() {
                span { {count_label(*n, "tool", "tools")} }
            }
            if let Backend::Online { version, .. } = &*backend.read() {
                span { class: "r", "p2a {version}" }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_read_naturally() {
        assert_eq!(count_label(0, "dataset", "datasets"), "0 datasets");
        assert_eq!(count_label(1, "dataset", "datasets"), "1 dataset");
        assert_eq!(count_label(270, "tool", "tools"), "270 tools");
    }

    #[test]
    fn without_a_provider_the_bar_says_so() {
        let settings = Settings::default();
        assert_eq!(model_label(&settings), "no model connected");
    }
}
