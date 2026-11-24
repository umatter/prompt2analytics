//! Dataset sidebar component for displaying loaded datasets and reload functionality

use dioxus::prelude::*;

use crate::api::{DatasetMeta, ReloadResult, api};
use crate::components::DatasetInspectorModal;
use crate::state::{AddDataRequest, DatasetInfo, SessionState};

/// Dataset sidebar component
#[component]
pub fn DatasetSidebar() -> Element {
    let mut session_state = use_context::<Signal<SessionState>>();
    let mut datasets = use_signal(Vec::<DatasetMeta>::new);
    let mut is_loading = use_signal(|| false);
    let mut is_reloading = use_signal(|| false);
    let add_data = use_context::<AddDataRequest>().0;
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
                        // The session's datasets as the server lists them, so
                        // the status bar counts them after a reload too.
                        let known: Vec<DatasetInfo> = ds
                            .iter()
                            .map(|d| DatasetInfo {
                                name: d.name.clone(),
                                rows: usize::try_from(d.row_count).unwrap_or(0),
                                cols: usize::try_from(d.column_count).unwrap_or(0),
                            })
                            .collect();
                        if session_state.peek().datasets_differ(&known) {
                            session_state.write().loaded_datasets = known;
                        }
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
        div { class: "h-full flex flex-col k-paper",
            // Header, as tall as the chat header
            div { class: "flex-shrink-0 h-[60px] px-[18px] k-rule-b",
                div { class: "h-full w-full flex items-center justify-between",
                    h2 { class: "text-[22px] font-extrabold k-cond", "Datasets" }
                    // Reload button
                    button {
                        class: "k-icon-btn disabled:opacity-30 disabled:cursor-not-allowed",
                        disabled: reloading || datasets_list.is_empty(),
                        title: "Reload all datasets from their source files",
                        "aria-label": "Reload all datasets from their source files",
                        onclick: reload_datasets,
                        if reloading {
                            svg {
                                class: "w-[18px] h-[18px] animate-spin",
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
                                class: "w-[18px] h-[18px]",
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
            div { class: "flex-1 overflow-y-auto",
                // Error display
                if let Some(ref e) = err {
                    div { class: "m-[18px] mb-0 pl-3 py-1 border-l-4 border-[var(--k-orange)] text-sm", role: "alert",
                        "{e}"
                    }
                }

                // Reload result display
                if let Some(ref res) = result {
                    div { class: "m-[18px] mb-0 pl-3 py-1 border-l-4 border-[var(--k-rule)] text-sm",
                        if !res.succeeded.is_empty() {
                            div { class: "mb-1",
                                "Reloaded: {res.succeeded.join(\", \")}"
                            }
                        }
                        if !res.failed.is_empty() {
                            div { class: "mb-1 k-orange-text",
                                "Failed: {res.failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(\", \")}"
                            }
                        }
                        if !res.skipped.is_empty() {
                            div { class: "k-muted",
                                "Skipped: {res.skipped.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(\", \")}"
                            }
                        }
                    }
                }

                // Loading state
                if loading {
                    div { class: "m-[18px] k-mono text-xs k-muted",
                        span { class: "k-orange k-spin" }
                        " loading datasets"
                    }
                } else if datasets_list.is_empty() {
                    // Empty state
                    div { class: "m-[18px] text-[13px] k-muted",
                        div { class: "k-mono text-[11.5px] k-ink border border-dashed border-[var(--k-rule)] px-4 py-[22px] mb-2.5",
                            "no data yet"
                        }
                        p { "Add a CSV, Parquet, Stata or SAS file, or load an example from the welcome screen. Data stays in this session." }
                        button {
                            class: "k-btn-pri mt-3",
                            onclick: move |_| {
                                let mut request = add_data;
                                *request.write() += 1;
                            },
                            "Add data"
                        }
                    }
                } else {
                    // Dataset list
                    div {
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

    let file_size = dataset
        .file_size_bytes
        .map(super::dataset_inspector::size_label);

    rsx! {
        div { class: "k-line-b",
            // Header
            button {
                class: "w-full px-[18px] pt-3.5 pb-1 flex items-center justify-between text-left hover:bg-[var(--k-fill)]",
                onclick: move |_| {
                    let current = *is_expanded.read();
                    is_expanded.set(!current);
                },
                div { class: "flex items-center gap-2 min-w-0",
                    span { class: "k-orange text-xs", "aria-hidden": "true", "●" }
                    span { class: "k-mono text-[13px] font-bold truncate",
                        "{dataset.name}"
                    }
                }
                // Expand icon
                svg {
                    class: "w-4 h-4 k-muted flex-shrink-0",
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
                div { class: "px-[18px] pb-3.5 pt-1 text-[12.5px]",
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
                        span { class: "k-mono",
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

                    // Columns (collapsible list). The count is the dataset's
                    // width: a wide dataset's row keeps its first names only.
                    if dataset.width() > 0 {
                        div { class: "mt-2 pt-2 border-t border-gray-100 dark:border-gray-700",
                            span { class: "text-gray-500 dark:text-gray-400 text-xs block mb-1",
                                "Columns ({dataset.width()}):"
                            }
                            div { class: "flex flex-wrap gap-1",
                                for col in dataset.column_names.iter().take(10) {
                                    span {
                                        key: "{col}",
                                        class: "k-mono text-[11.5px] pr-2",
                                        "{col}"
                                    }
                                }
                                // A row can hold no names (a first name too
                                // long to keep, or no column list).
                                if !dataset.has_column_names() {
                                    span { class: "px-1.5 py-0.5 text-xs text-gray-500 dark:text-gray-400",
                                        "names not stored"
                                    }
                                } else if dataset.columns_beyond(10) > 0 {
                                    span { class: "px-1.5 py-0.5 text-xs text-gray-500 dark:text-gray-400",
                                        "+{dataset.columns_beyond(10)} more"
                                    }
                                }
                            }
                        }
                    }

                    // Inspect button
                    div { class: "mt-3 pt-2 border-t border-gray-100 dark:border-gray-700",
                        button {
                            class: "k-btn w-full justify-center",
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
                            "Inspect columns"
                        }
                    }
                }
            }
        }
    }
}
