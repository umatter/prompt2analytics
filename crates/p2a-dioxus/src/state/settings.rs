//! Settings state management with platform-agnostic persistence
//!
//! - Web: Uses localStorage via gloo_storage
//! - Native: Uses file-based storage in config directory
//!
//! On native platforms (desktop/mobile), API keys are also read from environment
//! variables (OPENAI_API_KEY, ANTHROPIC_API_KEY) if not already set.

use crate::api::ProviderConfig;
use crate::platform::{StorageBackend, create_storage, is_native};
use serde::{Deserialize, Serialize};

/// Storage key for settings
const SETTINGS_KEY: &str = "p2a-settings";

/// Theme preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    System,
    #[default]
    Light,
    Dark,
}

impl Theme {
    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Theme::System => "System",
            Theme::Light => "Light",
            Theme::Dark => "Dark",
        }
    }
}

/// OpenRouter exposes an OpenAI-compatible endpoint; we route traffic through
/// the backend's OpenAI client with this base URL.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Provider types supported. `None` is the first-run default so that users are
/// prompted to choose a provider and enter a key before the first chat, rather
/// than silently hitting an unconfigured Ollama endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    None,
    Ollama,
    Anthropic,
    Openai,
    Openrouter,
}

impl Provider {
    /// Get the provider type string for API
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::None => "none",
            Provider::Ollama => "ollama",
            Provider::Anthropic => "anthropic",
            Provider::Openai => "openai",
            Provider::Openrouter => "openrouter",
        }
    }

    /// Human-readable display name for UI surfaces.
    pub fn display_name(&self) -> &'static str {
        match self {
            Provider::None => "Not connected",
            Provider::Ollama => "Ollama",
            Provider::Anthropic => "Anthropic",
            Provider::Openai => "OpenAI",
            Provider::Openrouter => "OpenRouter",
        }
    }

    /// Check if this provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        matches!(
            self,
            Provider::Anthropic | Provider::Openai | Provider::Openrouter
        )
    }

    /// Whether the user has actually picked a provider (vs. first-run default).
    pub fn is_configured(&self) -> bool {
        !matches!(self, Provider::None)
    }
}

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Theme preference
    #[serde(default)]
    pub theme: Theme,

    /// Current provider
    pub provider: Provider,

    // Ollama settings
    pub ollama_base_url: String,
    pub ollama_model: String,

    // Anthropic settings
    #[serde(default)]
    pub anthropic_api_key: String,
    pub anthropic_model: String,

    // OpenAI settings
    #[serde(default)]
    pub openai_api_key: String,
    pub openai_model: String,

    // OpenRouter settings (routed through the OpenAI-compatible backend client)
    #[serde(default)]
    pub openrouter_api_key: String,
    #[serde(default = "default_openrouter_model")]
    pub openrouter_model: String,

    // Common settings
    pub temperature: f64,
    pub max_tokens: u32,

    // Feature toggles
    pub interpret_results: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: Theme::Light,
            // First-run default: no provider, so the chat path prompts the
            // user to configure one instead of silently failing on Ollama.
            provider: Provider::None,
            ollama_base_url: "http://localhost:11434".to_string(),
            ollama_model: "llama3.2".to_string(),
            anthropic_api_key: String::new(),
            anthropic_model: "claude-sonnet-4-20250514".to_string(),
            openai_api_key: String::new(),
            openai_model: "gpt-4o".to_string(),
            openrouter_api_key: String::new(),
            openrouter_model: default_openrouter_model(),
            temperature: 0.7,
            max_tokens: 4096,
            interpret_results: true,
        }
    }
}

fn default_openrouter_model() -> String {
    "openai/gpt-4o-mini".to_string()
}

impl Settings {
    /// Load settings from platform-appropriate storage.
    /// On native platforms, also checks environment variables for API keys.
    pub fn load() -> Self {
        let storage = create_storage();
        let mut settings = match storage.get::<Self>(SETTINGS_KEY) {
            Ok(settings) => {
                tracing::info!("Loaded settings from storage");
                settings
            }
            Err(e) => {
                tracing::info!("No saved settings found, using defaults: {}", e);
                Self::default()
            }
        };

        // On native platforms, populate API keys from environment variables if not set
        if is_native() {
            settings.populate_from_env();
        }

        settings
    }

    /// Populate API keys from environment variables if not already set.
    /// This only works on native platforms (desktop/mobile).
    #[cfg(not(target_arch = "wasm32"))]
    fn populate_from_env(&mut self) {
        tracing::info!("Checking environment variables for API keys...");

        // Check OPENAI_API_KEY
        if self.openai_api_key.is_empty() {
            match std::env::var("OPENAI_API_KEY") {
                Ok(key) if !key.is_empty() => {
                    tracing::info!(
                        "Detected OPENAI_API_KEY from environment ({} chars)",
                        key.len()
                    );
                    self.openai_api_key = key;
                }
                Ok(_) => {
                    tracing::info!("OPENAI_API_KEY env var is empty");
                }
                Err(e) => {
                    tracing::info!("OPENAI_API_KEY not found in environment: {}", e);
                }
            }
        } else {
            tracing::info!("OPENAI_API_KEY already set in settings, skipping env check");
        }

        // Check ANTHROPIC_API_KEY
        if self.anthropic_api_key.is_empty() {
            match std::env::var("ANTHROPIC_API_KEY") {
                Ok(key) if !key.is_empty() => {
                    tracing::info!(
                        "Detected ANTHROPIC_API_KEY from environment ({} chars)",
                        key.len()
                    );
                    self.anthropic_api_key = key;
                }
                Ok(_) => {
                    tracing::info!("ANTHROPIC_API_KEY env var is empty");
                }
                Err(e) => {
                    tracing::info!("ANTHROPIC_API_KEY not found in environment: {}", e);
                }
            }
        } else {
            tracing::info!("ANTHROPIC_API_KEY already set in settings, skipping env check");
        }

        // Check OPENROUTER_API_KEY
        if self.openrouter_api_key.is_empty()
            && let Ok(key) = std::env::var("OPENROUTER_API_KEY")
            && !key.is_empty()
        {
            tracing::info!(
                "Detected OPENROUTER_API_KEY from environment ({} chars)",
                key.len()
            );
            self.openrouter_api_key = key;
        }
    }

    /// No-op on web platform
    #[cfg(target_arch = "wasm32")]
    fn populate_from_env(&mut self) {
        // Environment variables not available on web
    }

    /// Save settings to platform-appropriate storage
    pub fn save(&self) {
        let storage = create_storage();
        if let Err(e) = storage.set(SETTINGS_KEY, self) {
            tracing::error!("Failed to save settings: {}", e);
        } else {
            tracing::info!("Settings saved to storage");
        }
    }

    /// Convert to ProviderConfig for API requests. `None` when no provider
    /// has been configured yet — call sites should open the settings modal
    /// in that case rather than sending an invalid request.
    pub fn to_provider_config(&self) -> Option<ProviderConfig> {
        match self.provider {
            Provider::None => None,
            Provider::Ollama => Some(ProviderConfig {
                provider_type: "ollama".to_string(),
                api_key: None,
                base_url: Some(self.ollama_base_url.clone()),
                model: self.ollama_model.clone(),
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
            }),
            Provider::Anthropic => Some(ProviderConfig {
                provider_type: "anthropic".to_string(),
                api_key: if self.anthropic_api_key.is_empty() {
                    None
                } else {
                    Some(self.anthropic_api_key.clone())
                },
                base_url: None,
                model: self.anthropic_model.clone(),
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
            }),
            Provider::Openai => Some(ProviderConfig {
                provider_type: "openai".to_string(),
                api_key: if self.openai_api_key.is_empty() {
                    None
                } else {
                    Some(self.openai_api_key.clone())
                },
                base_url: None,
                model: self.openai_model.clone(),
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
            }),
            // OpenRouter is an OpenAI-compatible endpoint, so we reuse the
            // backend's OpenAI client with a custom base URL.
            Provider::Openrouter => Some(ProviderConfig {
                provider_type: "openai".to_string(),
                api_key: if self.openrouter_api_key.is_empty() {
                    None
                } else {
                    Some(self.openrouter_api_key.clone())
                },
                base_url: Some(OPENROUTER_BASE_URL.to_string()),
                model: self.openrouter_model.clone(),
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
            }),
        }
    }

    /// Get current model based on provider
    pub fn current_model(&self) -> &str {
        match self.provider {
            Provider::None => "",
            Provider::Ollama => &self.ollama_model,
            Provider::Anthropic => &self.anthropic_model,
            Provider::Openai => &self.openai_model,
            Provider::Openrouter => &self.openrouter_model,
        }
    }

    /// Set current model based on provider
    pub fn set_current_model(&mut self, model: &str) {
        match self.provider {
            Provider::None => {}
            Provider::Ollama => self.ollama_model = model.to_string(),
            Provider::Anthropic => self.anthropic_model = model.to_string(),
            Provider::Openai => self.openai_model = model.to_string(),
            Provider::Openrouter => self.openrouter_model = model.to_string(),
        }
    }

    /// Get current API key based on provider (empty for Ollama / None)
    pub fn current_api_key(&self) -> &str {
        match self.provider {
            Provider::None | Provider::Ollama => "",
            Provider::Anthropic => &self.anthropic_api_key,
            Provider::Openai => &self.openai_api_key,
            Provider::Openrouter => &self.openrouter_api_key,
        }
    }

    /// Set current API key based on provider
    pub fn set_current_api_key(&mut self, key: &str) {
        match self.provider {
            Provider::None | Provider::Ollama => {}
            Provider::Anthropic => self.anthropic_api_key = key.to_string(),
            Provider::Openai => self.openai_api_key = key.to_string(),
            Provider::Openrouter => self.openrouter_api_key = key.to_string(),
        }
    }

    /// Validate settings
    pub fn is_valid(&self) -> bool {
        if !self.provider.is_configured() {
            return false;
        }

        // Check if API key is required and present
        if self.provider.requires_api_key() {
            let key = self.current_api_key();
            if key.is_empty() {
                return false;
            }
        }

        // Check model is set
        !self.current_model().is_empty()
    }

    /// Get validation error message if any
    pub fn validation_error(&self) -> Option<String> {
        if !self.provider.is_configured() {
            return Some("Choose an LLM provider in Settings to start chatting.".to_string());
        }

        if self.provider.requires_api_key() && self.current_api_key().is_empty() {
            return Some(format!("{:?} requires an API key", self.provider));
        }

        if self.current_model().is_empty() {
            return Some("Model is required".to_string());
        }

        None
    }
}
