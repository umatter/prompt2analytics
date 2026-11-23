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
                    Ok(saved_to) => {
                        status_message.set(Some(match saved_to {
                            // Desktop: the full path of the written file
                            Some(path) => format!("Saved to {path}"),
                            None => format!("Downloaded {filename}"),
                        }));
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
    // Escape closes the open menu.
    let close_menu = use_callback(move |()| is_open.set(false));
    crate::app::use_escape_close(*is_open.read(), close_menu);

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
                class: "k-btn",
                disabled: !has_completed_messages,
                onclick: handle_toggle,
                title: if has_completed_messages {
                    "Export conversation"
                } else {
                    "No messages to export"
                },
                "Export"
                span { class: "k-mono text-[10px]", aria_hidden: "true", "\u{25BE}" }
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
                    class: "k-menu absolute right-0 top-full mt-1 w-60 z-20",
                    button {
                        class: "k-menu-item",
                        onclick: move |_| do_export(ExportFormat::Json),
                        span { "Download as JSON" }
                        span { class: "ext", ".json" }
                    }
                    button {
                        class: "k-menu-item",
                        onclick: move |_| do_export(ExportFormat::Markdown),
                        span { "Download as Markdown" }
                        span { class: "ext", ".md" }
                    }
                    button {
                        class: "k-menu-item",
                        onclick: move |_| do_export(ExportFormat::Html),
                        span { "Download as HTML report" }
                        span { class: "ext", ".html" }
                    }

                    div { class: "k-menu-sep" }

                    button {
                        class: "k-menu-item",
                        onclick: do_copy_markdown,
                        span { "Copy as Markdown" }
                        span { class: "ext", "clipboard" }
                    }
                }
            }

            // Status message toast
            if let Some(ref msg) = *status_message.read() {
                div {
                    class: "k-toast absolute right-0 top-full mt-1 z-30",
                    "{msg}"
                }
            }
        }
    }
}
