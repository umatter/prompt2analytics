//! Dataset sidebar component for displaying loaded datasets and reload functionality

use dioxus::prelude::*;

use crate::api::{DatasetMeta, ReloadResult, api};
use crate::components::{DatasetInspectorModal, P2aIconMinimal};
use crate::state::SessionState;

/// Dataset sidebar component
#[component]
pub fn DatasetSidebar() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let mut datasets = use_signal(Vec::<DatasetMeta>::new);
    let mut is_loading = use_signal(|| false);
    let mut is_reloading = use_signal(|| false);
    let mut error = use_signal(|| None::<String>);
    let mut reload_result = use_signal(|| None::<ReloadResult>);
    let mut selected_dataset = use_signal(|| None::<DatasetMeta>);

    // Load datasets when session changes or refresh is triggered
    use_effect(move || {
        let state = session_state.read();
        let session_id = state.session_id.clone();
        let _refresh_counter = state.datasets_refresh_counter; // Track this to trigger re-runs
        drop(state);

        if let Some(sid) = session_id {
            spawn(async move {
                is_loading.set(true);
                error.set(None);

                let client = api();
                match client.list_session_datasets(&sid).await {
                    Ok(ds) => {
                        datasets.set(ds);
                    }
                    Err(e) => {
                        error.set(Some(e));
                    }
                }
                is_loading.set(false);
            });
        }
    });

    // Reload datasets handler
    let reload_datasets = move |_| {
        let session_id = session_state.read().session_id.clone();
        if let Some(sid) = session_id {
            spawn(async move {
                is_reloading.set(true);
                error.set(None);
                reload_result.set(None);

                let client = api();
                match client.reload_session_datasets(&sid).await {
                    Ok(result) => {
                        reload_result.set(Some(result.clone()));

                        // Refresh dataset list
                        if let Ok(ds) = client.list_session_datasets(&sid).await {
                            datasets.set(ds);
                        }
                    }
                    Err(e) => {
                        error.set(Some(e));
                    }
                }
                is_reloading.set(false);
            });
        }
    };

    let datasets_list = datasets.read();
    let loading = *is_loading.read();
    let reloading = *is_reloading.read();
    let err = error.read().clone();
    let result = reload_result.read().clone();

    rsx! {
        div { class: "h-full flex flex-col bg-white dark:bg-gray-900 border-l border-gray-300 dark:border-gray-800",
            // Header - fixed height matching ChatPanel
            div { class: "flex-shrink-0 h-16 px-6 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900",
                div { class: "h-full w-full flex items-center justify-between",
                    div { class: "flex items-center gap-2",
                        P2aIconMinimal { size: 20.0 }
                        h1 { class: "text-xl font-bold text-gray-900 dark:text-white",
                            "Datasets"
                        }
                    }
                    // Reload button
                    button {
                        class: "p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors disabled:opacity-50",
                        disabled: reloading || datasets_list.is_empty(),
                        title: "Reload all datasets from source files",
                        onclick: reload_datasets,
                        if reloading {
                            svg {
                                class: "w-5 h-5 text-gray-600 dark:text-gray-400 animate-spin",
                                fill: "none",
                                view_box: "0 0 24 24",
                                circle {
                                    class: "opacity-25",
                                    cx: "12",
                                    cy: "12",
                                    r: "10",
                                    stroke: "currentColor",
                                    stroke_width: "4"
                                }
                                path {
                                    class: "opacity-75",
                                    fill: "currentColor",
                                    d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                }
                            }
                        } else {
                            svg {
                                class: "w-5 h-5 text-gray-600 dark:text-gray-400",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                }
                            }
                        }
                    }
                }
            }

            // Content
            div { class: "flex-1 overflow-y-auto p-4",
                // Error display
                if let Some(ref e) = err {
                    div { class: "mb-4 p-3 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-sm",
                        "{e}"
                    }
                }

                // Reload result display
                if let Some(ref res) = result {
                    div { class: "mb-4 p-3 rounded-lg bg-teal-50 dark:bg-teal-900/30 text-sm",
                        if !res.succeeded.is_empty() {
                            div { class: "text-green-700 dark:text-green-400 mb-1",
                                "Reloaded: {res.succeeded.join(\", \")}"
                            }
                        }
                        if !res.failed.is_empty() {
                            div { class: "text-red-700 dark:text-red-400 mb-1",
                                "Failed: {res.failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(\", \")}"
                            }
                        }
                        if !res.skipped.is_empty() {
                            div { class: "text-gray-600 dark:text-gray-400",
                                "Skipped: {res.skipped.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(\", \")}"
                            }
                        }
                    }
                }

                // Loading state
                if loading {
                    div { class: "flex items-center justify-center py-8",
                        svg {
                            class: "w-6 h-6 text-gray-400 animate-spin",
                            fill: "none",
                            view_box: "0 0 24 24",
                            circle {
                                class: "opacity-25",
                                cx: "12",
                                cy: "12",
                                r: "10",
                                stroke: "currentColor",
                                stroke_width: "4"
                            }
                            path {
                                class: "opacity-75",
                                fill: "currentColor",
                                d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            }
                        }
                    }
                } else if datasets_list.is_empty() {
                    // Empty state
                    div { class: "text-center py-8 px-2 text-gray-500 dark:text-gray-400",
                        svg {
                            class: "w-8 h-8 mx-auto mb-3 text-gray-300 dark:text-gray-600",
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
                        p { class: "break-words", "No datasets loaded" }
                        p { class: "text-sm mt-1 break-words", "Load a dataset using the chat" }
                    }
                } else {
                    // Dataset list
                    div { class: "space-y-3",
                        for dataset in datasets_list.iter() {
                            DatasetCard {
                                key: "{dataset.id}",
                                dataset: dataset.clone(),
                                on_inspect: move |ds: DatasetMeta| {
                                    selected_dataset.set(Some(ds));
                                }
                            }
                        }
                    }
                }
            }

            // Dataset Inspector Modal
            if let Some(ref ds) = *selected_dataset.read() {
                DatasetInspectorModal {
                    dataset: ds.clone(),
                    on_close: move |_| {
                        selected_dataset.set(None);
                    }
                }
            }
        }
    }
}

/// Props for DatasetCard
#[derive(Props, Clone, PartialEq)]
struct DatasetCardProps {
    dataset: DatasetMeta,
    on_inspect: EventHandler<DatasetMeta>,
}

/// Individual dataset card
#[component]
fn DatasetCard(props: DatasetCardProps) -> Element {
    let mut is_expanded = use_signal(|| false);
    let expanded = *is_expanded.read();
    let dataset = &props.dataset;
    let dataset_for_inspect = props.dataset.clone();

    // Format file size
    let file_size = dataset.file_size_bytes.map(|bytes| {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    });

    rsx! {
        div { class: "rounded-lg border border-gray-300 dark:border-gray-700 overflow-hidden bg-white dark:bg-gray-800",
            // Header
            button {
                class: "w-full px-3 py-2 flex items-center justify-between bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                onclick: move |_| {
                    let current = *is_expanded.read();
                    is_expanded.set(!current);
                },
                div { class: "flex items-center gap-2 min-w-0",
                    // Database icon
                    svg {
                        class: "w-4 h-4 text-teal-500 flex-shrink-0",
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
                    span { class: "font-medium text-gray-900 dark:text-white truncate",
                        "{dataset.name}"
                    }
                }
                // Expand icon
                svg {
                    class: "w-4 h-4 text-gray-400 flex-shrink-0 transition-transform duration-200",
                    class: if expanded { "rotate-180" } else { "" },
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "2",
                        d: "M19 9l-7 7-7-7"
                    }
                }
            }

            // Details (expandable)
            if expanded {
                div { class: "px-3 py-2 text-sm border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800",
                    // Dimensions
                    div { class: "flex items-center gap-2 mb-1",
                        span { class: "text-gray-500 dark:text-gray-400", "Size:" }
                        span { class: "text-gray-900 dark:text-white",
                            "{dataset.row_count} rows × {dataset.column_count} cols"
                        }
                    }

                    // File type
                    div { class: "flex items-center gap-2 mb-1",
                        span { class: "text-gray-500 dark:text-gray-400", "Type:" }
                        span { class: "text-gray-900 dark:text-white uppercase",
                            "{dataset.source_type}"
                        }
                    }

                    // File size
                    if let Some(ref size) = file_size {
                        div { class: "flex items-center gap-2 mb-1",
                            span { class: "text-gray-500 dark:text-gray-400", "File:" }
                            span { class: "text-gray-900 dark:text-white",
                                "{size}"
                            }
                        }
                    }

                    // Source path
                    if let Some(ref path) = dataset.source_path {
                        div { class: "mt-2 pt-2 border-t border-gray-100 dark:border-gray-700",
                            span { class: "text-gray-500 dark:text-gray-400 text-xs block mb-1", "Source:" }
                            span { class: "text-gray-600 dark:text-gray-300 text-xs font-mono break-all",
                                "{path}"
                            }
                        }
                    }

                    // Columns (collapsible list)
                    if !dataset.column_names.is_empty() {
                        div { class: "mt-2 pt-2 border-t border-gray-100 dark:border-gray-700",
                            span { class: "text-gray-500 dark:text-gray-400 text-xs block mb-1",
                                "Columns ({dataset.column_names.len()}):"
                            }
                            div { class: "flex flex-wrap gap-1",
                                for col in dataset.column_names.iter().take(10) {
                                    span {
                                        key: "{col}",
                                        class: "px-1.5 py-0.5 text-xs bg-gray-200 dark:bg-gray-700 rounded text-gray-700 dark:text-gray-300",
                                        "{col}"
                                    }
                                }
                                if dataset.column_names.len() > 10 {
                                    span { class: "px-1.5 py-0.5 text-xs text-gray-500 dark:text-gray-400",
                                        "+{dataset.column_names.len() - 10} more"
                                    }
                                }
                            }
                        }
                    }

                    // Inspect button
                    div { class: "mt-3 pt-2 border-t border-gray-100 dark:border-gray-700",
                        button {
                            class: "w-full px-3 py-1.5 text-xs font-medium text-teal-700 dark:text-teal-300 bg-teal-50 dark:bg-teal-900/30 rounded-lg hover:bg-teal-100 dark:hover:bg-teal-900/50 transition-colors flex items-center justify-center gap-1.5",
                            onclick: {
                                let dataset = dataset_for_inspect.clone();
                                move |evt: Event<MouseData>| {
                                    evt.stop_propagation();
                                    props.on_inspect.call(dataset.clone());
                                }
                            },
                            svg {
                                class: "w-3.5 h-3.5",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                                }
                            }
                            "View Details"
                        }
                    }
                }
            }
        }
    }
}
