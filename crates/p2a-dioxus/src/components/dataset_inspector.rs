//! Dataset inspector modal component

use dioxus::prelude::*;

use crate::api::DatasetMeta;

/// Props for DatasetInspectorModal
#[derive(Props, Clone, PartialEq)]
pub struct DatasetInspectorModalProps {
    /// The dataset to inspect
    pub dataset: DatasetMeta,
    /// Callback when modal should close
    pub on_close: EventHandler<()>,
}

/// A file size in binary units, as the server counts them (powers of 1024): `B`, `KiB` and
/// `MiB` with one decimal, `GiB` with two.
pub(crate) fn size_label(bytes: i64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < KIB * KIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < KIB * KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else {
        format!("{:.2} GiB", b / (KIB * KIB * KIB))
    }
}

/// Dataset inspector modal - shows detailed dataset information
#[component]
pub fn DatasetInspectorModal(props: DatasetInspectorModalProps) -> Element {
    let dataset = &props.dataset;
    let mut copied = use_signal(|| false);

    let file_size = dataset.file_size_bytes.map(size_label);
    let loaded = crate::utils::local_stamp(&dataset.loaded_at);
    // The first rows, fetched with head_dataset when asked for (None: not asked yet).
    let mut preview = use_signal(|| None::<Result<String, String>>);
    let session = use_context::<Signal<crate::state::SessionState>>();
    let show_preview = {
        let name = dataset.name.clone();
        move |_| {
            let name = name.clone();
            let Some(sid) = session.peek().get_session_id().map(str::to_string) else {
                preview.set(Some(Err("No session yet".into())));
                return;
            };
            preview.set(Some(Ok(String::new())));
            spawn(async move {
                let args = serde_json::json!({ "dataset": name, "n": 5 });
                let result = crate::api::ApiClient::new()
                    .call_tool(&sid, "head_dataset", args)
                    .await
                    .and_then(|r| {
                        let text: Vec<String> = r
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                crate::api::types::ContentItem::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect();
                        if r.success {
                            Ok(text.join("\n"))
                        } else {
                            Err(r.error.unwrap_or_else(|| text.join("\n")))
                        }
                    });
                preview.set(Some(result));
            });
        }
    };

    // Handle copy dataset name
    let handle_copy_name = {
        let name = dataset.name.clone();
        move |_| {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(window) = web_sys::window() {
                    let navigator = window.navigator();
                    let clipboard = navigator.clipboard();
                    let name_clone = name.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let _ =
                            wasm_bindgen_futures::JsFuture::from(clipboard.write_text(&name_clone))
                                .await;
                    });
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                // Escape the name so a value containing quotes/backslashes cannot
                // break out of the JS string literal and run arbitrary code.
                let js = format!(
                    "navigator.clipboard.writeText('{}')",
                    crate::utils::js_single_quoted(&name)
                );
                dioxus::document::eval(&js);
            }
            copied.set(true);
            // Reset after 2 seconds
            spawn(async move {
                #[cfg(all(
                    not(target_arch = "wasm32"),
                    any(feature = "desktop", feature = "mobile")
                ))]
                {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    gloo_timers::future::TimeoutFuture::new(2_000).await;
                }
                copied.set(false);
            });
        }
    };

    rsx! {
        // Container for backdrop and modal
        div {
            class: "fixed inset-0 z-50 flex items-center justify-center p-4",

            // Backdrop
            div {
                class: "k-backdrop",
                onclick: move |_| props.on_close.call(()),
            }

            // Modal
            div {
                class: "k-modal max-w-lg max-h-[85vh]",

                // Header
                div { class: "k-modal-head",
                    div { class: "min-w-0",
                        h2 { class: "k-modal-title k-mono truncate", style: "font-stretch: 75%;",
                            "{dataset.name}"
                        }
                        p { class: "k-modal-sub", "Dataset" }
                    }
                    button {
                        class: "k-icon-btn flex-shrink-0",
                        title: "Close",
                        aria_label: "Close",
                        onclick: move |_| props.on_close.call(()),
                        svg {
                            class: "w-4 h-4",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "square",
                                stroke_width: "2",
                                d: "M6 18L18 6M6 6l12 12"
                            }
                        }
                    }
                }

                // Content
                div { class: "k-modal-body space-y-6",
                    // Overview
                    table { class: "k-table",
                        thead {
                            tr {
                                th { "Rows" }
                                th { "Columns" }
                                if file_size.is_some() {
                                    th { "File size" }
                                }
                            }
                        }
                        tbody {
                            tr {
                                td { "{dataset.row_count}" }
                                td { "{dataset.column_count}" }
                                if let Some(ref size) = file_size {
                                    td { "{size}" }
                                }
                            }
                        }
                    }

                    // Source
                    div {
                        h3 { class: "k-section-h", "Source" }
                        div { class: "k-row",
                            span { class: "k-muted", "Type" }
                            span { class: "k-tag", "{dataset.source_type}" }
                        }
                        if let Some(ref path) = dataset.source_path {
                            div { class: "k-row",
                                span { class: "k-muted flex-shrink-0", "Path" }
                                code { class: "k-mono text-[11.5px] break-all text-right", "{path}" }
                            }
                        }
                        div { class: "k-row",
                            span { class: "k-muted", "Loaded" }
                            span { class: "k-mono text-[11.5px]", "{loaded}" }
                        }
                    }

                    // Columns
                    div {
                        h3 { class: "k-section-h",
                            "Columns "
                            span { class: "k-mono k-muted font-normal text-[11.5px]", "{dataset.width()}" }
                        }
                        // A wide dataset's row keeps its first names only, and
                        // a row can hold none (a first name too long to keep,
                        // or a result without a column list).
                        if dataset.has_column_names() {
                            if dataset.column_names_cut() {
                                p { class: "k-help mb-1",
                                    "First {dataset.column_names.len()} of {dataset.width()} shown"
                                }
                            }
                            ol { class: "k-cols pt-1",
                                for (i, col) in dataset.column_names.iter().enumerate() {
                                    li { key: "{col}", title: "{col}",
                                        span { class: "ty", "{i + 1:>3}" }
                                        "{col}"
                                        if let Some(t) = dataset.column_types.get(i) {
                                            span { class: "k-mono k-muted text-[11px] ml-2", "{t}" }
                                        }
                                    }
                                }
                            }
                        } else if dataset.width() > 0 {
                            p { class: "k-help", "Column names not stored" }
                        }
                    }

                    // First rows, on request
                    if dataset.in_memory {
                        div {
                            h3 { class: "k-section-h", "First rows" }
                            match preview() {
                                None => rsx! {
                                    button { class: "k-btn", onclick: show_preview, "Show the first 5 rows" }
                                },
                                Some(Ok(ref t)) if t.is_empty() => rsx! {
                                    p { class: "k-help", span { class: "k-spin" } " loading" }
                                },
                                Some(Ok(ref t)) => rsx! {
                                    pre { class: "k-mono text-[11.5px] overflow-x-auto whitespace-pre", "{t}" }
                                },
                                Some(Err(ref e)) => rsx! {
                                    p { class: "k-help k-orange-text", role: "alert", "{e}" }
                                },
                            }
                        }
                    }
                }

                // Footer with actions
                div { class: "k-modal-foot justify-between",
                    button {
                        class: "k-btn",
                        onclick: handle_copy_name,
                        if *copied.read() {
                            span { class: "k-orange", "\u{2713}" }
                            "Copied"
                        } else {
                            "Copy name"
                        }
                    }
                    button {
                        class: "k-btn-pri",
                        onclick: move |_| props.on_close.call(()),
                        "Close"
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::size_label;

    #[test]
    fn file_sizes_are_labelled_in_binary_units() {
        assert_eq!(size_label(512), "512 B");
        assert_eq!(size_label(1536), "1.5 KiB");
        assert_eq!(size_label(3 << 19), "1.5 MiB");
        assert_eq!(size_label(3 << 29), "1.50 GiB");
        // A negative size from the server is shown as it is, not wrapped.
        assert_eq!(size_label(-1), "-1 B");
    }
}
