//! Export menu dropdown component

use dioxus::prelude::*;

use crate::export::{
    ExportFormat, HtmlExportOptions, JsonExportOptions, MarkdownExportOptions, copy_to_clipboard,
    export_to_html, export_to_json, export_to_markdown, generate_filename, trigger_download,
};
use crate::state::settings::Theme;
use crate::state::{ChatState, ConversationState, Settings};

/// Export menu dropdown component
#[component]
pub fn ExportMenu() -> Element {
    let chat_state = use_context::<Signal<ChatState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let settings = use_context::<Signal<Settings>>();

    let mut is_open = use_signal(|| false);
    let mut status_message = use_signal(|| Option::<String>::None);

    // Check if export is available (has completed messages)
    let has_completed_messages = chat_state
        .read()
        .messages
        .iter()
        .any(|m| !m.is_streaming && !m.content.is_empty());

    // Handle export action
    let mut do_export = move |format: ExportFormat| {
        let conversation = conversation_state
            .read()
            .get_current_conversation()
            .cloned();
        let messages = chat_state.read().messages.clone();
        let theme = settings.read().theme;

        // Generate content based on format
        let result = match format {
            ExportFormat::Json => {
                let options = JsonExportOptions {
                    pretty: true,
                    include_tool_results: true,
                    include_images: true,
                };
                export_to_json(conversation.as_ref(), &messages, &options)
            }
            ExportFormat::Markdown => {
                let options = MarkdownExportOptions::default();
                Ok(export_to_markdown(
                    conversation.as_ref(),
                    &messages,
                    &options,
                ))
            }
            ExportFormat::Html => {
                let options = HtmlExportOptions {
                    dark_theme: theme == Theme::Dark,
                    include_tool_calls: true,
                    include_images: true,
                };
                Ok(export_to_html(conversation.as_ref(), &messages, &options))
            }
        };

        match result {
            Ok(content) => {
                let filename =
                    generate_filename(conversation.as_ref().map(|c| c.title.as_str()), format);
                match trigger_download(&content, &filename, format) {
                    Ok(()) => {
                        status_message.set(Some(format!("Exported to {}", filename)));
                        // Clear status after 3 seconds
                        spawn(async move {
                            #[cfg(target_arch = "wasm32")]
                            {
                                gloo_timers::future::TimeoutFuture::new(3000).await;
                            }
                            #[cfg(all(
                                not(target_arch = "wasm32"),
                                any(feature = "desktop", feature = "mobile")
                            ))]
                            {
                                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                            }
                            status_message.set(None);
                        });
                    }
                    Err(e) => {
                        status_message.set(Some(format!("Export failed: {}", e)));
                    }
                }
            }
            Err(e) => {
                status_message.set(Some(format!("Export failed: {}", e)));
            }
        }

        is_open.set(false);
    };

    // Handle copy to clipboard
    let do_copy_markdown = move |_| {
        let conversation = conversation_state
            .read()
            .get_current_conversation()
            .cloned();
        let messages = chat_state.read().messages.clone();

        let options = MarkdownExportOptions::default();
        let content = export_to_markdown(conversation.as_ref(), &messages, &options);

        match copy_to_clipboard(&content) {
            Ok(()) => {
                status_message.set(Some("Copied to clipboard".to_string()));
                spawn(async move {
                    #[cfg(target_arch = "wasm32")]
                    {
                        gloo_timers::future::TimeoutFuture::new(2000).await;
                    }
                    #[cfg(all(
                        not(target_arch = "wasm32"),
                        any(feature = "desktop", feature = "mobile")
                    ))]
                    {
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                    status_message.set(None);
                });
            }
            Err(e) => {
                status_message.set(Some(format!("Copy failed: {}", e)));
            }
        }

        is_open.set(false);
    };

    // Close menu when clicking outside
    let handle_close = move |_| {
        is_open.set(false);
    };

    // Toggle open state
    let handle_toggle = move |_| {
        if has_completed_messages {
            let current = *is_open.read();
            is_open.set(!current);
        }
    };

    rsx! {
        div { class: "relative",
            // Export button
            button {
                class: if has_completed_messages {
                    "px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 dark:text-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors flex items-center gap-1.5"
                } else {
                    "px-3 py-1.5 text-sm font-medium text-gray-400 bg-gray-100 rounded-lg cursor-not-allowed dark:text-gray-500 dark:bg-gray-800 flex items-center gap-1.5"
                },
                disabled: !has_completed_messages,
                onclick: handle_toggle,
                title: if has_completed_messages {
                    "Export conversation"
                } else {
                    "No messages to export"
                },
                // Download icon
                svg {
                    class: "w-4 h-4",
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "2",
                        d: "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    }
                }
                "Export"
                // Dropdown arrow
                svg {
                    class: "w-3 h-3",
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

            // Dropdown menu
            if *is_open.read() {
                // Backdrop for closing
                div {
                    class: "fixed inset-0 z-10",
                    onclick: handle_close
                }

                // Menu
                div {
                    class: "absolute right-0 top-full mt-1 w-56 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20 py-1",
                    // JSON export
                    button {
                        class: "w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3",
                        onclick: move |_| do_export(ExportFormat::Json),
                        svg {
                            class: "w-4 h-4 text-gray-500 dark:text-gray-400",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"
                            }
                        }
                        span { "Download as JSON" }
                    }

                    // Markdown export
                    button {
                        class: "w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3",
                        onclick: move |_| do_export(ExportFormat::Markdown),
                        svg {
                            class: "w-4 h-4 text-gray-500 dark:text-gray-400",
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
                        span { "Download as Markdown" }
                    }

                    // HTML export
                    button {
                        class: "w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3",
                        onclick: move |_| do_export(ExportFormat::Html),
                        svg {
                            class: "w-4 h-4 text-gray-500 dark:text-gray-400",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                            }
                        }
                        span { "Download as HTML Report" }
                    }

                    // Divider
                    div { class: "my-1 border-t border-gray-200 dark:border-gray-700" }

                    // Copy as Markdown
                    button {
                        class: "w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3",
                        onclick: do_copy_markdown,
                        svg {
                            class: "w-4 h-4 text-gray-500 dark:text-gray-400",
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
                        span { "Copy as Markdown" }
                    }
                }
            }

            // Status message toast
            if let Some(ref msg) = *status_message.read() {
                div {
                    class: "absolute right-0 top-full mt-1 px-3 py-2 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm rounded-lg shadow-lg z-30 whitespace-nowrap",
                    "{msg}"
                }
            }
        }
    }
}
