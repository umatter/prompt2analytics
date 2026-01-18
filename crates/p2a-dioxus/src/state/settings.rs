//! Settings state management with platform-agnostic persistence
//!
//! - Web: Uses localStorage via gloo_storage
//! - Native: Uses file-based storage in config directory

use crate::api::ProviderConfig;
use crate::platform::{create_storage, StorageBackend};
use serde::{Deserialize, Serialize};

/// Storage key for settings
const SETTINGS_KEY: &str = "p2a-settings";

/// Provider types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Ollama,
    Anthropic,
    Openai,
}

impl Provider {
    /// Get the provider type string for API
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Ollama => "ollama",
            Provider::Anthropic => "anthropic",
            Provider::Openai => "openai",
        }
    }

    /// Check if this provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        matches!(self, Provider::Anthropic | Provider::Openai)
    }
}

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
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

    // Common settings
    pub temperature: f64,
    pub max_tokens: u32,

    // Feature toggles
    pub interpret_results: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: Provider::Ollama,
            ollama_base_url: "http://localhost:11434".to_string(),
            ollama_model: "llama3.2".to_string(),
            anthropic_api_key: String::new(),
            anthropic_model: "claude-sonnet-4-20250514".to_string(),
            openai_api_key: String::new(),
            openai_model: "gpt-4o".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
            interpret_results: true,
        }
    }
}

impl Settings {
    /// Load settings from platform-appropriate storage
    pub fn load() -> Self {
        let storage = create_storage();
        match storage.get::<Self>(SETTINGS_KEY) {
            Ok(settings) => {
                tracing::info!("Loaded settings from storage");
                settings
            }
            Err(e) => {
                tracing::info!("No saved settings found, using defaults: {}", e);
                Self::default()
            }
        }
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

    /// Convert to ProviderConfig for API requests
    pub fn to_provider_config(&self) -> ProviderConfig {
        match self.provider {
            Provider::Ollama => ProviderConfig {
                provider_type: "ollama".to_string(),
                api_key: None,
                base_url: Some(self.ollama_base_url.clone()),
                model: self.ollama_model.clone(),
                temperature: Some(self.temperature),
                max_tokens: Some(self.max_tokens),
            },
            Provider::Anthropic => ProviderConfig {
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
            },
            Provider::Openai => ProviderConfig {
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
            },
        }
    }

    /// Get current model based on provider
    pub fn current_model(&self) -> &str {
        match self.provider {
            Provider::Ollama => &self.ollama_model,
            Provider::Anthropic => &self.anthropic_model,
            Provider::Openai => &self.openai_model,
        }
    }

    /// Set current model based on provider
    pub fn set_current_model(&mut self, model: &str) {
        match self.provider {
            Provider::Ollama => self.ollama_model = model.to_string(),
            Provider::Anthropic => self.anthropic_model = model.to_string(),
            Provider::Openai => self.openai_model = model.to_string(),
        }
    }

    /// Get current API key based on provider (empty for Ollama)
    pub fn current_api_key(&self) -> &str {
        match self.provider {
            Provider::Ollama => "",
            Provider::Anthropic => &self.anthropic_api_key,
            Provider::Openai => &self.openai_api_key,
        }
    }

    /// Set current API key based on provider
    pub fn set_current_api_key(&mut self, key: &str) {
        match self.provider {
            Provider::Ollama => {}
            Provider::Anthropic => self.anthropic_api_key = key.to_string(),
            Provider::Openai => self.openai_api_key = key.to_string(),
        }
    }

    /// Validate settings
    pub fn is_valid(&self) -> bool {
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
        if self.provider.requires_api_key() && self.current_api_key().is_empty() {
            return Some(format!(
                "{:?} requires an API key",
                self.provider
            ));
        }

        if self.current_model().is_empty() {
            return Some("Model is required".to_string());
        }

        None
    }
}
