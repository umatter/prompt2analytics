//! Settings modal component for configuring LLM provider

use dioxus::events::FormData;
use dioxus::prelude::*;

use crate::api::ApiClient;
use crate::app::apply_theme;
use crate::state::SessionState;
use crate::state::settings::{Provider, Settings, Theme};

/// Parse dimensions from dataset load result message
fn parse_dimensions(msg: &str) -> (usize, usize) {
    // Format: "Dimensions: X rows x Y columns"
    if let Some(dims_start) = msg.find("Dimensions:") {
        let dims_part = &msg[dims_start..];
        if let Some(rows_end) = dims_part.find(" rows") {
            let rows_str = dims_part[11..rows_end].trim();
            if let Ok(rows) = rows_str.parse::<usize>()
                && let Some(cols_start) = dims_part.find("x ")
                && let Some(cols_end) = dims_part.find(" columns")
            {
                let cols_str = dims_part[cols_start + 2..cols_end].trim();
                if let Ok(cols) = cols_str.parse::<usize>() {
                    return (rows, cols);
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
    let mut local_theme = use_signal(|| settings.read().theme);
    let mut local_provider = use_signal(|| settings.read().provider);
    let mut local_api_key = use_signal(|| settings.read().current_api_key().to_string());
    let mut local_model = use_signal(|| settings.read().current_model().to_string());
    let mut local_temperature = use_signal(|| settings.read().temperature);
    let mut local_base_url = use_signal(|| settings.read().ollama_base_url.clone());

    // Data loading state
    let mut file_path = use_signal(String::new);
    let mut dataset_name = use_signal(String::new);
    let mut file_type = use_signal(|| "auto".to_string()); // auto, csv, parquet, excel, stata, sas
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
        local_theme.set(s.theme);
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
        s.theme = *local_theme.read();
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
            "openrouter" => Provider::Openrouter,
            _ => Provider::None,
        };
        local_provider.set(provider);

        // Update API key and model for this provider from settings
        let s = settings.read();
        let (api_key, model) = match provider {
            Provider::None => (String::new(), String::new()),
            Provider::Ollama => ("".to_string(), s.ollama_model.clone()),
            Provider::Anthropic => (s.anthropic_api_key.clone(), s.anthropic_model.clone()),
            Provider::Openai => (s.openai_api_key.clone(), s.openai_model.clone()),
            Provider::Openrouter => (s.openrouter_api_key.clone(), s.openrouter_model.clone()),
        };
        local_api_key.set(api_key);
        local_model.set(model);
    };

    let handle_theme_change = move |evt: Event<FormData>| {
        let value = evt.value();
        let theme = match value.as_str() {
            "system" => Theme::System,
            "light" => Theme::Light,
            "dark" => Theme::Dark,
            _ => Theme::System,
        };
        local_theme.set(theme);
        // Apply theme immediately for instant feedback
        apply_theme(theme);
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
        let selected_file_type = file_type.read().clone();
        let uploaded_file_name = selected_file_name.read().clone();
        let uploaded_file_content = selected_file_content.read().clone();

        // Determine if we're using uploaded file or path
        let use_upload = uploaded_file_content.is_some() && uploaded_file_name.is_some();

        if !use_upload && path.trim().is_empty() {
            load_error.set(Some(
                "Please select a file or enter a file path".to_string(),
            ));
            return;
        }

        let dataset_name_to_use = if name.trim().is_empty() {
            if use_upload {
                uploaded_file_name
                    .as_ref()
                    .map(|n| n.split('.').next().unwrap_or("dataset").to_string())
                    .unwrap_or_else(|| "dataset".to_string())
            } else {
                path.split('/')
                    .next_back()
                    .unwrap_or("dataset")
                    .split('.')
                    .next()
                    .unwrap_or("dataset")
                    .to_string()
            }
        } else {
            name
        };

        load_status.set(Some("Loading...".to_string()));
        load_error.set(None);

        let mut session = session_state;
        spawn(async move {
            // Ensure session exists (creates one if needed) WITHOUT holding a
            // signal borrow across `.await`. Holding `session.write()` across the
            // session-creation network call makes concurrent reads of
            // `session_state` (e.g. ChatPanel rendering `loaded_datasets`) panic
            // with `AlreadyBorrowedMut`. Mirrors the pattern used in ChatPanel.
            let existing_id = session.read().session_id.clone();
            let session_id = if let Some(id) = existing_id {
                id
            } else {
                let snapshot = session.read().clone();
                match snapshot.initialize().await {
                    Ok(id) => {
                        session.write().set_session_id(id.clone());
                        id
                    }
                    Err(e) => {
                        load_error.set(Some(format!("Session error: {}", e)));
                        load_status.set(None);
                        return;
                    }
                }
            };

            let client = ApiClient::new();

            let result = if use_upload {
                // Use upload_dataset for browser-selected files
                let mut args = serde_json::json!({
                    "content": uploaded_file_content.unwrap(),
                    "filename": uploaded_file_name.unwrap(),
                    "name": dataset_name_to_use
                });
                // Add file_type if not auto
                if selected_file_type != "auto" {
                    args["file_type"] = serde_json::Value::String(selected_file_type.clone());
                }
                client.call_tool(&session_id, "upload_dataset", args).await
            } else {
                // Use load_dataset for path-based loading
                let mut args = serde_json::json!({
                    "path": path,
                    "name": dataset_name_to_use
                });
                // Add file_type if not auto
                if selected_file_type != "auto" {
                    args["file_type"] = serde_json::Value::String(selected_file_type.clone());
                }
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
                        session
                            .write()
                            .add_dataset(dataset_name_to_use.clone(), rows, cols);
                    } else {
                        load_error
                            .set(Some(res.error.unwrap_or_else(|| "Load failed".to_string())));
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
    let current_theme = *local_theme.read();
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
                class: "relative bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-hidden border border-gray-200 dark:border-gray-800",

                // Header
                div { class: "px-6 py-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between",
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
                div { class: "px-6 pt-4 flex gap-2 border-b border-gray-200 dark:border-gray-800",
                    button {
                        class: if current_tab == "llm" {
                            "px-4 py-2 text-sm font-medium text-teal-600 dark:text-teal-400 border-b-2 border-teal-600 dark:border-teal-400 -mb-px"
                        } else {
                            "px-4 py-2 text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                        },
                        onclick: move |_| active_tab.set("llm".to_string()),
                        "LLM Provider"
                    }
                    button {
                        class: if current_tab == "data" {
                            "px-4 py-2 text-sm font-medium text-teal-600 dark:text-teal-400 border-b-2 border-teal-600 dark:border-teal-400 -mb-px"
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
                            // Theme selection
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Theme"
                                }
                                select {
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
                                    value: "{current_theme.as_str().to_lowercase()}",
                                    onchange: handle_theme_change,
                                    option { value: "system", "System (auto)" }
                                    option { value: "light", "Light" }
                                    option { value: "dark", "Dark" }
                                }
                            }

                            // First-run nudge: no provider configured yet.
                            if current_provider == Provider::None {
                                div { class: "p-3 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800",
                                    p { class: "text-sm text-teal-700 dark:text-teal-300",
                                        "Welcome! Choose an LLM provider below to start chatting. Anthropic and OpenAI need an API key; Ollama runs locally on your machine."
                                    }
                                }
                            }

                            // Provider selection
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Provider"
                                }
                                select {
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
                                    value: "{current_provider.as_str()}",
                                    onchange: handle_provider_change,
                                    option { value: "none", disabled: true, "— Select a provider —" }
                                    option { value: "anthropic", "Anthropic (Claude)" }
                                    option { value: "openai", "OpenAI (GPT)" }
                                    option { value: "openrouter", "OpenRouter (100+ models)" }
                                    option { value: "ollama", "Ollama (Local)" }
                                }
                                // Helper text per provider: where to get an API key.
                                if current_provider == Provider::Anthropic {
                                    p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1",
                                        "Get a key at console.anthropic.com/settings/keys"
                                    }
                                } else if current_provider == Provider::Openai {
                                    p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1",
                                        "Get a key at platform.openai.com/api-keys"
                                    }
                                } else if current_provider == Provider::Openrouter {
                                    p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1",
                                        "Get a key at openrouter.ai/keys — one key, 100+ models (model format: provider/model, e.g. openai/gpt-4o-mini)"
                                    }
                                }
                            }

                            // Ollama base URL
                            if current_provider == Provider::Ollama {
                                div {
                                    label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                        "Ollama Base URL"
                                    }
                                    input {
                                        class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
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
                                        class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
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
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
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
                                    class: "w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer accent-teal-600",
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
                            div { class: "p-3 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800",
                                p { class: "text-sm text-teal-700 dark:text-teal-300",
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
                                        class: "flex-1 px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer",
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
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent font-mono text-sm",
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

                            // File type selector
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "File Type"
                                }
                                select {
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
                                    value: "{file_type}",
                                    onchange: move |evt| file_type.set(evt.value().clone()),
                                    option { value: "auto", "Auto-detect (from extension)" }
                                    option { value: "csv", "CSV (.csv)" }
                                    option { value: "parquet", "Parquet (.parquet)" }
                                    option { value: "excel", "Excel (.xlsx, .xls)" }
                                    option { value: "stata", "Stata (.dta)" }
                                    option { value: "sas", "SAS (.sas7bdat)" }
                                }
                                p { class: "text-xs text-gray-500 dark:text-gray-400 mt-1",
                                    "Usually auto-detect works. Override if needed."
                                }
                            }

                            // Dataset name
                            div {
                                label { class: "block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5",
                                    "Dataset Name (optional)"
                                }
                                input {
                                    class: "w-full px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 focus:border-transparent",
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
                div { class: "px-6 py-4 border-t border-gray-200 dark:border-gray-800 flex justify-end gap-3",
                    button {
                        class: "px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors",
                        onclick: handle_cancel,
                        "Cancel"
                    }
                    if current_tab == "llm" {
                        button {
                            class: "px-4 py-2 text-sm font-medium text-white bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors",
                            onclick: handle_save,
                            "Save Settings"
                        }
                    } else {
                        button {
                            class: "px-4 py-2 text-sm font-medium text-white bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors",
                            onclick: handle_cancel,
                            "Done"
                        }
                    }
                }
            }
        }
    }
}
