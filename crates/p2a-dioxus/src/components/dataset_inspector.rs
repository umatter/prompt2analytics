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

/// Dataset inspector modal - shows detailed dataset information
#[component]
pub fn DatasetInspectorModal(props: DatasetInspectorModalProps) -> Element {
    let dataset = &props.dataset;
    let mut copied = use_signal(|| false);

    // Format file size
    let file_size = dataset.file_size_bytes.map(|bytes| {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    });

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
                        let _ = wasm_bindgen_futures::JsFuture::from(
                            clipboard.write_text(&name_clone)
                        ).await;
                    });
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let js = format!("navigator.clipboard.writeText('{}')", name);
                dioxus::document::eval(&js);
            }
            copied.set(true);
            // Reset after 2 seconds
            spawn(async move {
                #[cfg(not(target_arch = "wasm32"))]
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
                class: "absolute inset-0 bg-black/50 backdrop-blur-sm",
                onclick: move |_| props.on_close.call(()),
            }

            // Modal
            div {
                class: "relative bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-lg max-h-[85vh] overflow-hidden border border-gray-200 dark:border-gray-800",

                // Header
                div { class: "px-6 py-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between",
                    div { class: "flex items-center gap-3 min-w-0",
                        // Database icon
                        div { class: "w-10 h-10 rounded-lg bg-teal-100 dark:bg-teal-900/30 flex items-center justify-center flex-shrink-0",
                            svg {
                                class: "w-5 h-5 text-teal-600 dark:text-teal-400",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"
                                }
                            }
                        }
                        div { class: "min-w-0",
                            h2 { class: "text-xl font-semibold text-gray-900 dark:text-white truncate",
                                "{dataset.name}"
                            }
                            p { class: "text-sm text-gray-500 dark:text-gray-400",
                                "Dataset Inspector"
                            }
                        }
                    }
                    button {
                        class: "p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex-shrink-0",
                        onclick: move |_| props.on_close.call(()),
                        svg {
                            class: "w-5 h-5",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M6 18L18 6M6 6l12 12"
                            }
                        }
                    }
                }

                // Content
                div { class: "px-6 py-4 overflow-y-auto max-h-[60vh]",
                    // Overview section
                    div { class: "mb-6",
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Overview"
                        }
                        div { class: "grid grid-cols-2 gap-4",
                            // Dimensions
                            StatCard {
                                label: "Dimensions",
                                value: format!("{} x {}", dataset.row_count, dataset.column_count),
                                icon: rsx! {
                                    svg {
                                        class: "w-4 h-4",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M4 6h16M4 12h16M4 18h16"
                                        }
                                    }
                                }
                            }
                            // Rows
                            StatCard {
                                label: "Rows",
                                value: format!("{}", dataset.row_count),
                                icon: rsx! {
                                    svg {
                                        class: "w-4 h-4",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M4 6h16M4 10h16M4 14h16M4 18h16"
                                        }
                                    }
                                }
                            }
                            // Columns
                            StatCard {
                                label: "Columns",
                                value: format!("{}", dataset.column_count),
                                icon: rsx! {
                                    svg {
                                        class: "w-4 h-4",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
                                        }
                                    }
                                }
                            }
                            // File size
                            if let Some(ref size) = file_size {
                                StatCard {
                                    label: "File Size",
                                    value: size.clone(),
                                    icon: rsx! {
                                        svg {
                                            class: "w-4 h-4",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Source info
                    div { class: "mb-6",
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Source"
                        }
                        div { class: "bg-gray-50 dark:bg-gray-800 rounded-lg p-3 space-y-2",
                            // Type
                            div { class: "flex items-center justify-between",
                                span { class: "text-sm text-gray-500 dark:text-gray-400", "Type" }
                                span { class: "px-2 py-0.5 text-xs font-medium rounded bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300 uppercase",
                                    "{dataset.source_type}"
                                }
                            }
                            // Path
                            if let Some(ref path) = dataset.source_path {
                                div {
                                    span { class: "text-sm text-gray-500 dark:text-gray-400 block mb-1", "Path" }
                                    code { class: "text-xs font-mono text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded block break-all",
                                        "{path}"
                                    }
                                }
                            }
                            // Loaded at
                            div { class: "flex items-center justify-between",
                                span { class: "text-sm text-gray-500 dark:text-gray-400", "Loaded" }
                                span { class: "text-sm text-gray-700 dark:text-gray-300",
                                    "{dataset.loaded_at}"
                                }
                            }
                        }
                    }

                    // Columns section
                    div {
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Columns ({dataset.column_names.len()})"
                        }
                        div { class: "bg-gray-50 dark:bg-gray-800 rounded-lg p-3",
                            div { class: "flex flex-wrap gap-2",
                                for col in dataset.column_names.iter() {
                                    span {
                                        key: "{col}",
                                        class: "inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-300",
                                        // Column icon
                                        svg {
                                            class: "w-3 h-3 text-gray-400",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M7 20l4-16m2 16l4-16M6 9h14M4 15h14"
                                            }
                                        }
                                        "{col}"
                                    }
                                }
                            }
                        }
                    }
                }

                // Footer with actions
                div { class: "px-6 py-4 border-t border-gray-200 dark:border-gray-800 flex items-center justify-between",
                    // Copy name button
                    button {
                        class: "px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2",
                        onclick: handle_copy_name,
                        if *copied.read() {
                            svg {
                                class: "w-4 h-4 text-green-600 dark:text-green-400",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M5 13l4 4L19 7"
                                }
                            }
                            "Copied!"
                        } else {
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                                }
                            }
                            "Copy Name"
                        }
                    }
                    // Close button
                    button {
                        class: "px-4 py-2 text-sm font-medium text-white bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors",
                        onclick: move |_| props.on_close.call(()),
                        "Close"
                    }
                }
            }
        }
    }
}

/// Props for StatCard
#[derive(Props, Clone, PartialEq)]
struct StatCardProps {
    label: &'static str,
    value: String,
    icon: Element,
}

/// A stat card showing a label, value, and icon
#[component]
fn StatCard(props: StatCardProps) -> Element {
    rsx! {
        div { class: "bg-gray-50 dark:bg-gray-800 rounded-lg p-3",
            div { class: "flex items-center gap-2 mb-1",
                span { class: "text-gray-400 dark:text-gray-500",
                    {props.icon}
                }
                span { class: "text-xs text-gray-500 dark:text-gray-400",
                    "{props.label}"
                }
            }
            span { class: "text-lg font-semibold text-gray-900 dark:text-white",
                "{props.value}"
            }
        }
    }
}
