//! The welcome screen of an empty conversation: the long logo, a row of
//! facts about the backend, one line on what p2a is, and the bundled example
//! datasets as starters.

use dioxus::prelude::*;

use crate::api::ApiClient;
use crate::components::P2aLockup;
use crate::examples::{EXAMPLES, Example, load_example};
use crate::state::{ConnectModelRequest, SessionState, Settings};

/// Props for [`Welcome`].
#[derive(Props, Clone, PartialEq)]
pub struct WelcomeProps {
    /// Sends a prompt (the chat panel's send).
    #[props(default)]
    pub on_suggestion: Option<EventHandler<String>>,
    /// The starters are inactive while a send runs.
    #[props(default)]
    pub disabled: bool,
}

/// The welcome screen.
#[component]
pub fn Welcome(props: WelcomeProps) -> Element {
    let session = use_context::<Signal<SessionState>>();
    let settings = use_context::<Signal<Settings>>();
    let mut loading = use_signal(|| None::<&'static str>);
    let mut error = use_signal(|| None::<String>);
    // An example loaded while no model was connected: its question waits
    // here and is asked as soon as a model is set up.
    let mut waiting = use_signal(|| None::<Example>);
    let mut connect_model = use_context::<ConnectModelRequest>().0;
    let on_suggestion = props.on_suggestion;
    use_effect(move || {
        if settings.read().provider.is_configured()
            && let Some(example) = waiting()
        {
            waiting.set(None);
            if let Some(handler) = on_suggestion {
                handler.call(example.prompt.to_string());
            }
        }
    });
    // Backend version and tool count for the tag row (None until known).
    let mut facts = use_signal(|| None::<(String, Option<usize>)>);
    use_future(move || async move {
        let client = ApiClient::new();
        if let Ok(h) = client.health().await {
            let tools = client.list_tools().await.ok().map(|t| t.len());
            facts.set(Some((h.version, tools)));
        }
    });

    let mut start = move |example: Example| {
        if loading.peek().is_some() {
            return;
        }
        loading.set(Some(example.name));
        error.set(None);
        waiting.set(None);
        spawn(async move {
            match load_example(session, example).await {
                Ok(()) => {
                    // With a provider set up, ask the starter's question;
                    // otherwise the data waits in the list and the question
                    // waits for a model.
                    if !settings.peek().provider.is_configured() {
                        waiting.set(Some(example));
                    } else if let Some(handler) = on_suggestion {
                        handler.call(example.prompt.to_string());
                    }
                }
                Err(e) => error.set(Some(format!("Could not load {}: {e}", example.name))),
            }
            loading.set(None);
        });
    };

    rsx! {
        div { class: "px-12 pt-11 pb-6 max-w-[920px]",
            P2aLockup { cell: 10.0 }
            div { class: "flex flex-wrap gap-1.5 mt-6",
                if let Some((ref version, tools)) = *facts.read() {
                    if let Some(n) = tools {
                        Tag { label: "tools", value: n.to_string() }
                    }
                    Tag { label: "p2a", value: version.clone() }
                }
                Tag { label: "license", value: "MIT/Apache-2.0".to_string() }
                Tag { label: "engine", value: "pure Rust".to_string() }
            }
            p { class: "text-[15px] max-w-[66ch] mt-4 mb-8",
                "Chat-first data analytics: validated, pre-implemented econometrics, statistics, machine learning and charts behind one conversation."
            }
            p { class: "text-sm k-muted max-w-[60ch] mb-3",
                "Ask for an analysis in plain words. Every number comes from a tool call you can open, check and export as a script."
            }
            div { class: "k-rule-t",
                for example in EXAMPLES {
                    button {
                        key: "{example.name}",
                        class: "k-start",
                        disabled: props.disabled || loading.read().is_some(),
                        onclick: move |_| start(example),
                        span { class: "n", "{example.name}" }
                        span {
                            "{example.title}"
                            span { class: "d", "{example.question}" }
                        }
                        span { class: "go",
                            if *loading.read() == Some(example.name) {
                                span { class: "k-spin" }
                                " loading"
                            } else {
                                "load"
                            }
                        }
                    }
                }
            }
            if let Some(example) = waiting() {
                div {
                    class: "k-notice mt-4 flex flex-col items-start gap-2",
                    role: "status",
                    span {
                        b { "{example.name} is loaded." }
                        " Connect a model to ask the question; p2a asks it as soon as one is set up."
                    }
                    button {
                        class: "k-btn-pri",
                        onclick: move |_| connect_model += 1,
                        "Connect a model"
                    }
                }
            }
            if let Some(ref msg) = *error.read() {
                p { class: "mt-3 text-sm k-orange-text", role: "alert", "{msg}" }
            }
            p { class: "mt-3 text-[12.5px] k-muted",
                "The examples are small public datasets bundled with p2a. Or add your own: CSV, Parquet, Stata or SAS."
            }
        }
    }
}

/// A label and a value in a small outlined box, like a README badge.
#[component]
fn Tag(label: &'static str, value: String) -> Element {
    rsx! {
        span { class: "k-mono text-[11.5px] border border-[var(--k-rule)] px-1.5 py-[1px]",
            "{label} "
            span { class: "k-orange-text font-semibold", "{value}" }
        }
    }
}
