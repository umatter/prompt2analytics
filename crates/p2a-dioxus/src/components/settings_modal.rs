//! Settings modal component for configuring LLM provider

use dioxus::prelude::*;
use dioxus::events::FormData;

use crate::api::ApiClient;
use crate::state::settings::{Provider, Settings};
use crate::state::SessionState;

/// Parse dimensions from dataset load result message
fn parse_dimensions(msg: &str) -> (usize, usize) {
    // Format: "Dimensions: X rows x Y columns"
    if let Some(dims_start) = msg.find("Dimensions:") {
        let dims_part = &msg[dims_start..];
        if let Some(rows_end) = dims_part.find(" rows") {
            let rows_str = dims_part[11..rows_end].trim();
            if let Ok(rows) = rows_str.parse::<usize>() {
                if let Some(cols_start) = dims_part.find("x ") {
                    if let Some(cols_end) = dims_part.find(" columns") {
                        let cols_str = dims_part[cols_start + 2..cols_end].trim();
                        if let Ok(cols) = cols_str.parse::<usize>() {
                            return (rows, cols);
                        }
                    }
                }
            }
        }
    }
    (0, 0)
}

/// Props for SettingsModal
#[derive(Props, Clone, PartialEq)]
pub struct SettingsModalProps {
    /// Whether the modal is open
    pub is_open: bool,
    /// Initial tab to show ("llm" or "data")
    #[props(default = "llm".to_string())]
    pub initial_tab: String,
    /// Callback when modal should close
    pub on_close: EventHandler<()>,
}

/// Settings modal component
#[component]
pub fn SettingsModal(props: SettingsModalProps) -> Element {
    let mut settings = use_context::<Signal<Settings>>();
    let session_state = use_context::<Signal<SessionState>>();

    // Local state for editing
    let mut local_provider = use_signal(|| settings.read().provider);
    let mut local_api_key = use_signal(|| settings.read().current_api_key().to_string());
    let mut local_model = use_signal(|| settings.read().current_model().to_string());
    let mut local_temperature = use_signal(|| settings.read().temperature);
    let mut local_base_url = use_signal(|| settings.read().ollama_base_url.clone());

    // Data loading state
    let mut file_path = use_signal(String::new);
    let mut dataset_name = use_signal(String::new);
    let mut load_status = use_signal(|| None::<String>);
    let mut load_error = use_signal(|| None::<String>);

    // File upload state (for browser file picker)
    let mut selected_file_name = use_signal(|| None::<String>);
    let mut selected_file_content = use_signal(|| None::<String>); // base64 encoded

    // Current tab - always start with the prop value
    let mut active_tab = use_signal(|| "llm".to_string());

    // Track previous open state to detect when modal opens
    let mut was_open = use_signal(|| false);

    // Sync local state when modal opens (transition from closed to open)
    if props.is_open && !*was_open.read() {
        // Modal just opened - reset to initial tab and sync settings
        active_tab.set(props.initial_tab.clone());
        let s = settings.read();
        local_provider.set(s.provider);
        local_api_key.set(s.current_api_key().to_string());
        local_model.set(s.current_model().to_string());
        local_temperature.set(s.temperature);
        local_base_url.set(s.ollama_base_url.clone());
    }

    // Update tracking state
    if props.is_open != *was_open.read() {
        was_open.set(props.is_open);
    }

    if !props.is_open {
        return rsx! {};
    }

    let handle_save = move |_| {
        let mut s = settings.write();
        s.provider = *local_provider.read();
        s.set_current_api_key(&local_api_key.read());
        s.set_current_model(&local_model.read());
        s.temperature = *local_temperature.read();
        if s.provider == Provider::Ollama {
            s.ollama_base_url = local_base_url.read().clone();
        }
        s.save();
        props.on_close.call(());
    };

    let handle_cancel = move |_| {
        props.on_close.call(());
    };

    let handle_provider_change = move |evt: Event<FormData>| {
        let value = evt.value();
        let provider = match value.as_str() {
            "ollama" => Provider::Ollama,
            "anthropic" => Provider::Anthropic,
            "openai" => Provider::Openai,
            _ => Provider::Ollama,
        };
        local_provider.set(provider);

        // Update model to default for this provider
        let default_model = match provider {
            Provider::Ollama => "llama3.2",
            Provider::Anthropic => "claude-sonnet-4-20250514",
            Provider::Openai => "gpt-4o",
        };
        local_model.set(default_model.to_string());
    };

    // Handle file selection from browser file picker
    let handle_file_select = move |evt: Event<FormData>| {
        let files = evt.files();
        if let Some(file) = files.into_iter().next() {
            let file_name = file.name();
            selected_file_name.set(Some(file_name.clone()));
            load_status.set(Some("Reading file...".to_string()));
            load_error.set(None);

            spawn(async move {
                match file.read_bytes().await {
                    Ok(bytes) => {
                        use base64::{Engine as _, engine::general_purpose::STANDARD};
                        let base64_content = STANDARD.encode(&bytes);
                        selected_file_content.set(Some(base64_content));
                        load_status.set(Some(format!("Selected: {}", file_name)));
                    }
                    Err(e) => {
                        load_error.set(Some(format!("Failed to read file: {}", e)));
                        load_status.set(None);
                    }
                }
            });
        }
    };

    // Handle load dataset
    let handle_load_dataset = move |_| {
        let path = file_path.read().clone();
        let name = dataset_name.read().clone();
        let uploaded_file_name = selected_file_name.read().clone();
        let uploaded_file_content = selected_file_content.read().clone();

        // Determine if we're using uploaded file or path
        let use_upload = uploaded_file_content.is_some() && uploaded_file_name.is_some();

        if !use_upload && path.trim().is_empty() {
            load_error.set(Some("Please select a file or enter a file path".to_string()));
            return;
        }

        let dataset_name_to_use = if name.trim().is_empty() {
            if use_upload {
                uploaded_file_name.as_ref()
                    .map(|n| n.split('.').next().unwrap_or("dataset").to_string())
                    .unwrap_or_else(|| "dataset".to_string())
            } else {
                path.split('/').last().unwrap_or("dataset").split('.').next().unwrap_or("dataset").to_string()
            }
        } else {
            name
        };

        load_status.set(Some("Loading...".to_string()));
        load_error.set(None);

        let mut session = session_state;
        spawn(async move {
            // Ensure session exists (creates one if needed)
            let session_id = match session.write().ensure_session().await {
                Ok(id) => id,
                Err(e) => {
                    load_error.set(Some(format!("Session error: {}", e)));
                    load_status.set(None);
                    return;
                }
            };

            let client = ApiClient::new();

            let result = if use_upload {
                // Use upload_dataset for browser-selected files
                let args = serde_json::json!({
                    "content": uploaded_file_content.unwrap(),
                    "filename": uploaded_file_name.unwrap(),
                    "name": dataset_name_to_use
                });
                client.call_tool(&session_id, "upload_dataset", args).await
            } else {
                // Use load_dataset for path-based loading
                let args = serde_json::json!({
                    "path": path,
                    "name": dataset_name_to_use
                });
                client.call_tool(&session_id, "load_dataset", args).await
            };

            match result {
                Ok(res) => {
                    if res.success {
                        load_status.set(Some(format!("Loaded: {}", dataset_name_to_use)));
                        load_error.set(None);
                        file_path.set(String::new());
                        dataset_name.set(String::new());
                        selected_file_name.set(None);
                        selected_file_content.set(None);

                        // Parse rows/cols from result message (format: "Dimensions: X rows x Y columns")
                        let (rows, cols) = {
                            // Find first text content
                            let text = res.content.iter().find_map(|item| {
                                if let crate::api::types::ContentItem::Text { text } = item {
                                    Some(text.as_str())
                                } else {
                                    None
                                }
                            });
                            if let Some(msg) = text {
                                parse_dimensions(msg)
                            } else {
                                (0, 0)
                            }
                        };

                        // Add to session state
                        session.write().add_dataset(dataset_name_to_use.clone(), rows, cols);
                    } else {
                        load_error.set(Some(res.error.unwrap_or_else(|| "Load failed".to_string())));
                        load_status.set(None);
                    }
                }
                Err(e) => {
                    load_error.set(Some(e));
                    load_status.set(None);
                }
            }
        });
    };

    let current_provider = *local_provider.read();
    let requires_api_key = current_provider.requires_api_key();
    let current_tab = active_tab.read().clone();

    rsx! {
        // Container for backdrop and modal (siblings for clean event handling)
        div {
            class: "fixed inset-0 z-50 flex items-center justify-center p-4",

            // Backdrop - separate clickable layer behind modal
            div {
                class: "absolute inset-0 bg-black/50 backdrop-blur-sm",
                onclick: move |_| props.on_close.call(()),
            }

            // Modal - positioned above backdrop, clicks don't propagate to backdrop
            div {
                class: "relative bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-hidden",

                // Header
                div { class: "px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between",
                    h2 { class: "text-xl font-semibold text-gray-900 dark:text-white", "Settings" }
                    button {
                        class: "p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
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

                // Tabs
                div { class: "px-6 pt-4 flex gap-2 border-b border-gray-200 dark:border-gray-700",
                    button {
                        class: if current_tab == "llm" {
                            "px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 -mb-px"
                        } else {
                            "px-4 py-2 text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                        },
                        onclick: move |_| active_tab.set("llm".to_string()),
                        "LLM Provider"
                    }
                    button {
                        class: if current_tab == "data" {
                            "px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 -mb-px"
                        } else {
                            "px-4 py-2 text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                        },
                        onclick: move |_| active_tab.set("data".to_string()),
                        "Load Data"
                    }
                }

                // Content
                div { class: "px-6 py-4 overflow-y-auto max-h-[60vh]",
                    if current_tab == "llm" {
                        // LLM Settings
                        div { class: "space-y-4",
                            // Provider selection
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Provider"
                                }
                                select {
                                    class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                    value: "{current_provider.as_str()}",
                                    onchange: handle_provider_change,
                                    option { value: "ollama", "Ollama (Local)" }
                                    option { value: "anthropic", "Anthropic (Claude)" }
                                    option { value: "openai", "OpenAI (GPT)" }
                                }
                            }

                            // Ollama base URL
                            if current_provider == Provider::Ollama {
                                div {
                                    label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                        "Ollama Base URL"
                                    }
                                    input {
                                        class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                        r#type: "text",
                                        placeholder: "http://localhost:11434",
                                        value: "{local_base_url}",
                                        oninput: move |evt| local_base_url.set(evt.value().clone())
                                    }
                                }
                            }

                            // API Key
                            if requires_api_key {
                                div {
                                    label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                        "API Key"
                                    }
                                    input {
                                        class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                        r#type: "password",
                                        placeholder: "Enter your API key...",
                                        value: "{local_api_key}",
                                        oninput: move |evt| local_api_key.set(evt.value().clone())
                                    }
                                    p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1.5 flex items-center gap-1",
                                        svg {
                                            class: "w-3.5 h-3.5",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                                            }
                                        }
                                        "API keys are stored locally and never sent to our servers."
                                    }
                                }
                            }

                            // Model
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Model"
                                }
                                input {
                                    class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                    r#type: "text",
                                    value: "{local_model}",
                                    oninput: move |evt| local_model.set(evt.value().clone())
                                }
                            }

                            // Temperature
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Temperature: {local_temperature:.1}"
                                }
                                input {
                                    class: "w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-600",
                                    r#type: "range",
                                    min: "0",
                                    max: "2",
                                    step: "0.1",
                                    value: "{local_temperature}",
                                    oninput: move |evt| {
                                        if let Ok(val) = evt.value().parse::<f64>() {
                                            local_temperature.set(val);
                                        }
                                    }
                                }
                                div { class: "flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1",
                                    span { "Precise" }
                                    span { "Creative" }
                                }
                            }
                        }
                    } else {
                        // Data Loading
                        div { class: "space-y-4",
                            // Info box
                            div { class: "p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800",
                                p { class: "text-sm text-blue-700 dark:text-blue-300",
                                    "Load datasets from your local filesystem. Supported formats: CSV, Parquet, Excel, Stata (.dta), SAS (.sas7bdat)"
                                }
                            }

                            // Browse Files section
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Browse Files"
                                }
                                div { class: "flex gap-2",
                                    // Hidden file input
                                    input {
                                        class: "hidden",
                                        r#type: "file",
                                        id: "file-upload-input",
                                        accept: ".csv,.parquet,.xlsx,.xls,.xlsb,.ods,.dta,.sas7bdat",
                                        onchange: handle_file_select
                                    }
                                    // Browse button
                                    label {
                                        r#for: "file-upload-input",
                                        class: "flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer",
                                        svg {
                                            class: "w-5 h-5",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                                            }
                                        }
                                        "Browse Files..."
                                    }
                                }
                                // Show selected file
                                if let Some(ref fname) = *selected_file_name.read() {
                                    div { class: "mt-2 px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800",
                                        p { class: "text-sm text-green-700 dark:text-green-300 flex items-center gap-2",
                                            svg {
                                                class: "w-4 h-4 flex-shrink-0",
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
                                            span { class: "truncate", "{fname}" }
                                        }
                                    }
                                }
                            }

                            // Divider
                            div { class: "flex items-center gap-3",
                                div { class: "flex-1 h-px bg-gray-200 dark:bg-gray-700" }
                                span { class: "text-xs text-gray-500 dark:text-gray-400", "OR" }
                                div { class: "flex-1 h-px bg-gray-200 dark:bg-gray-700" }
                            }

                            // File path (manual entry)
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Enter File Path"
                                }
                                input {
                                    class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm",
                                    r#type: "text",
                                    placeholder: "/path/to/your/data.csv",
                                    value: "{file_path}",
                                    oninput: move |evt| {
                                        file_path.set(evt.value().clone());
                                        // Clear uploaded file when manual path is entered
                                        if !evt.value().is_empty() {
                                            selected_file_name.set(None);
                                            selected_file_content.set(None);
                                        }
                                    }
                                }
                                p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1",
                                    "Use this for files on the server or when Browse doesn't work."
                                }
                            }

                            // Dataset name
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Dataset Name (optional)"
                                }
                                input {
                                    class: "w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                    r#type: "text",
                                    placeholder: "Auto-generated from filename",
                                    value: "{dataset_name}",
                                    oninput: move |evt| dataset_name.set(evt.value().clone())
                                }
                            }

                            // Load button
                            button {
                                class: "w-full px-4 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2",
                                onclick: handle_load_dataset,
                                svg {
                                    class: "w-5 h-5",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                                    }
                                }
                                "Load Dataset"
                            }

                            // Status/Error
                            if let Some(ref status) = *load_status.read() {
                                div { class: "p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800",
                                    p { class: "text-sm text-green-700 dark:text-green-300 flex items-center gap-2",
                                        svg {
                                            class: "w-4 h-4",
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
                                        "{status}"
                                    }
                                }
                            }

                            if let Some(ref error) = *load_error.read() {
                                div { class: "p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800",
                                    p { class: "text-sm text-red-700 dark:text-red-300 flex items-center gap-2",
                                        svg {
                                            class: "w-4 h-4",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                            }
                                        }
                                        "{error}"
                                    }
                                }
                            }
                        }
                    }
                }

                // Footer
                div { class: "px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3",
                    button {
                        class: "px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors",
                        onclick: handle_cancel,
                        "Cancel"
                    }
                    if current_tab == "llm" {
                        button {
                            class: "px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors",
                            onclick: handle_save,
                            "Save Settings"
                        }
                    } else {
                        button {
                            class: "px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors",
                            onclick: handle_cancel,
                            "Done"
                        }
                    }
                }
            }
        }
    }
}
