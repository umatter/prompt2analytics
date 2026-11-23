//! The welcome screen of an empty conversation: "Grüezi" in block letters,
//! the greeting in five languages, and the bundled example datasets as
//! starters.

use dioxus::prelude::*;

use crate::examples::{EXAMPLES, Example, load_example};
use crate::state::{SessionState, Settings};

/// Block letters of "Grüezi", one string per row; `#` is a filled cell and
/// `o` an orange one (the umlaut dots).
const GRUEZI: [&[&str]; 6] = [
    &[
        " ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### ",
    ],
    &["    ", "    ", "# ##", "##  ", "#   ", "#   ", "#   "],
    &[
        " o o ", "     ", "#   #", "#   #", "#   #", "#  ##", " ## #",
    ],
    &[
        "     ", "     ", " ### ", "#   #", "#####", "#    ", " ####",
    ],
    &[
        "     ", "     ", "#####", "   # ", "  #  ", " #   ", "#####",
    ],
    &[" # ", "   ", "## ", " # ", " # ", " # ", "###"],
];

/// Cell size of the block letters, in pixels.
const CELL: u32 = 11;

/// The greetings next to "Grüezi", in the order shown.
const GREETINGS: [&str; 5] = ["Welcome", "Bienvenue", "Benvenuto", "Willkommen", "欢迎"];

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

    let mut start = move |example: Example| {
        if loading.peek().is_some() {
            return;
        }
        loading.set(Some(example.name));
        error.set(None);
        let on_suggestion = props.on_suggestion;
        spawn(async move {
            match load_example(session, example).await {
                Ok(()) => {
                    // With a provider set up, ask the starter's question;
                    // otherwise the data is loaded and waits in the list.
                    if settings.peek().provider.is_configured()
                        && let Some(handler) = on_suggestion
                    {
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
            BlockWord {}
            div { class: "flex flex-wrap gap-x-[22px] gap-y-1 mt-[18px] mb-[30px] text-base k-muted",
                for (i, greeting) in GREETINGS.iter().enumerate() {
                    span { class: if i == 0 { "k-ink" } else { "" }, "{greeting}" }
                }
            }
            p { class: "text-base max-w-[60ch] mb-4",
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
            if let Some(ref msg) = *error.read() {
                p { class: "mt-3 text-sm k-orange-text", role: "alert", "{msg}" }
            }
            p { class: "mt-3 text-[12.5px] k-muted",
                "The examples are small public datasets bundled with p2a. Or add your own: CSV, Parquet, Stata or SAS."
            }
        }
    }
}

/// "Grüezi" drawn from square cells.
#[component]
fn BlockWord() -> Element {
    rsx! {
        div { class: "k-big", role: "img", "aria-label": "Grüezi",
            for (n, rows) in GRUEZI.iter().enumerate() {
                span {
                    key: "{n}",
                    class: "ch",
                    style: "grid-template-columns: repeat({rows[0].len()}, {CELL}px); grid-auto-rows: {CELL}px;",
                    for (r, row) in rows.iter().enumerate() {
                        for (c, cell) in row.chars().enumerate() {
                            i {
                                key: "{r}-{c}",
                                class: match cell {
                                    '#' => "on",
                                    'o' => "u",
                                    _ => "",
                                },
                                style: "width: {CELL}px; height: {CELL}px;",
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_letter_has_seven_rows_of_one_width() {
        for rows in GRUEZI {
            assert_eq!(rows.len(), 7);
            assert!(
                rows.iter()
                    .all(|r| r.chars().count() == rows[0].chars().count())
            );
        }
    }
}
