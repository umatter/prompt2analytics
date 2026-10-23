//! Settings state management with platform-agnostic persistence
//!
//! - Web: Uses localStorage via gloo_storage
//! - Native: Uses a file in the user's config directory, written owner-only
//!   (mode 0600 in a 0700 directory on Unix)
//!
//! On native platforms (desktop/mobile), API keys are also read from the
//! environment variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and
//! `OPENROUTER_API_KEY` when no key was typed in. Keys from the environment
//! are used but never saved: [`Settings::save`] persists
//! [`Settings::to_persisted`], which drops them. A copy that an older version
//! wrote to the settings file is removed when the settings are loaded.

use crate::api::ProviderConfig;
use crate::platform::{StorageBackend, create_storage, is_native};
use serde::{Deserialize, Serialize};

/// Storage key for settings
const SETTINGS_KEY: &str = "p2a-settings";

/// Providers that take an API key.
const KEYED_PROVIDERS: [Provider; 3] =
    [Provider::Anthropic, Provider::Openai, Provider::Openrouter];

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

    /// The environment variable the native app reads this provider's API
    /// key from, if any.
    pub fn env_var(&self) -> Option<&'static str> {
        match self {
            Provider::None | Provider::Ollama => None,
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::Openai => Some("OPENAI_API_KEY"),
            Provider::Openrouter => Some("OPENROUTER_API_KEY"),
        }
    }
}

/// Where an API key in [`Settings`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeySource {
    /// No key.
    #[default]
    Unset,
    /// Typed into the app (or loaded from storage): persisted on save.
    User,
    /// Read from the provider's environment variable: never persisted.
    Environment,
}

/// The source of each provider's API key. Not serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeySources {
    pub anthropic: KeySource,
    pub openai: KeySource,
    pub openrouter: KeySource,
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
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub anthropic_api_key: String,
    pub anthropic_model: String,

    // OpenAI settings
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openai_api_key: String,
    pub openai_model: String,

    // OpenRouter settings (routed through the OpenAI-compatible backend client)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub openrouter_api_key: String,
    #[serde(default = "default_openrouter_model")]
    pub openrouter_model: String,

    // Common settings
    pub temperature: f64,
    pub max_tokens: u32,

    // Feature toggles
    pub interpret_results: bool,

    /// Optional bearer token for a p2a-mcp server started with
    /// `--access-token` (`P2A_ACCESS_TOKEN`). Stored the same way as the API
    /// keys; sent only to the backend.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub backend_access_token: String,

    /// Where each API key came from; decides what [`Settings::to_persisted`]
    /// keeps.
    #[serde(skip)]
    pub key_sources: KeySources,
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
            backend_access_token: String::new(),
            key_sources: KeySources::default(),
        }
    }
}

fn default_openrouter_model() -> String {
    "openai/gpt-4o-mini".to_string()
}

impl Settings {
    /// Load settings from platform-appropriate storage.
    /// On native platforms, also checks environment variables for API keys.
    /// If the stored settings still hold a key equal to its environment
    /// variable (written by an older version), they are rewritten at once
    /// without it; see `load_from`.
    /// Publishes the backend access token for ApiClient and the chat stream.
    pub fn load() -> Self {
        let storage = create_storage();
        // On native platforms, populate API keys from environment variables if not set
        let env = is_native().then_some(|var: &str| std::env::var(var).ok());
        let settings = Self::load_from(&storage, env);

        // Publish the backend access token for ApiClient and the chat stream.
        crate::platform::set_backend_access_token(Some(&settings.backend_access_token));

        settings
    }

    /// Read the settings from `storage` and, when `env` is given, fill API
    /// keys from the environment through it.
    ///
    /// If that turns a stored key into an environment key (an older version
    /// wrote the environment key to disk in plain text), the stored copy is
    /// rewritten at once without it, rather than at the user's next save.
    fn load_from<S: StorageBackend>(
        storage: &S,
        env: Option<impl Fn(&str) -> Option<String>>,
    ) -> Self {
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

        settings.mark_stored_keys();

        if let Some(lookup) = env {
            tracing::info!("Checking environment variables for API keys...");
            if settings.populate_from_env_with(lookup) {
                tracing::info!("Rewriting settings without the stored environment key");
                settings.write_to(storage);
            }
        }

        settings
    }

    /// Mutable access to a provider's key and its source (`None` for
    /// providers without a key).
    fn key_slot(&mut self, provider: Provider) -> Option<(&mut String, &mut KeySource)> {
        match provider {
            Provider::None | Provider::Ollama => None,
            Provider::Anthropic => {
                Some((&mut self.anthropic_api_key, &mut self.key_sources.anthropic))
            }
            Provider::Openai => Some((&mut self.openai_api_key, &mut self.key_sources.openai)),
            Provider::Openrouter => Some((
                &mut self.openrouter_api_key,
                &mut self.key_sources.openrouter,
            )),
        }
    }

    /// Where the given provider's API key came from (`Unset` for providers
    /// without a key).
    pub fn key_source(&self, provider: Provider) -> KeySource {
        match provider {
            Provider::None | Provider::Ollama => KeySource::Unset,
            Provider::Anthropic => self.key_sources.anthropic,
            Provider::Openai => self.key_sources.openai,
            Provider::Openrouter => self.key_sources.openrouter,
        }
    }

    /// A copy fit for storage: identical, except that keys read from the
    /// environment are cleared, so they are never written anywhere.
    pub fn to_persisted(&self) -> Settings {
        let mut persisted = self.clone();
        for provider in KEYED_PROVIDERS {
            if let Some((key, source)) = persisted.key_slot(provider)
                && *source == KeySource::Environment
            {
                key.clear();
                *source = KeySource::Unset;
            }
        }
        persisted
    }

    /// Mark every non-empty key as user-provided (used right after loading
    /// from storage, before the environment is consulted).
    fn mark_stored_keys(&mut self) {
        for provider in KEYED_PROVIDERS {
            if let Some((key, source)) = self.key_slot(provider)
                && !key.is_empty()
            {
                *source = KeySource::User;
            }
        }
    }

    /// Fill API keys from environment variables, looked up through `lookup`.
    ///
    /// An empty key takes the environment value and is marked
    /// `Environment`. A stored key equal to the environment value is also
    /// marked `Environment`: that is a key an older version copied to disk,
    /// and any save now leaves it out. Any other key is left alone. Values are
    /// never logged, only their length.
    ///
    /// Returns true when a stored key was reclassified that way, i.e. when
    /// the stored settings still hold a key that must not be on disk.
    fn populate_from_env_with(&mut self, lookup: impl Fn(&str) -> Option<String>) -> bool {
        let mut reclassified = false;
        for provider in KEYED_PROVIDERS {
            let Some(var) = provider.env_var() else {
                continue;
            };
            let env = lookup(var).filter(|v| !v.is_empty());
            let Some((key, source)) = self.key_slot(provider) else {
                continue;
            };
            match env {
                Some(value) if key.is_empty() => {
                    tracing::info!("Detected {var} from environment ({} chars)", value.len());
                    *key = value;
                    *source = KeySource::Environment;
                }
                Some(value) if *key == value => {
                    tracing::info!("Stored key equals {var}; it will not be saved again");
                    reclassified |= *source == KeySource::User;
                    *source = KeySource::Environment;
                }
                Some(_) => {
                    tracing::info!("{var} is set, but the key from settings takes precedence");
                }
                None => {
                    tracing::info!("{var} not set in the environment");
                }
            }
        }
        reclassified
    }

    /// Save settings to platform-appropriate storage. Keys read from the
    /// environment are not written. Also publishes the backend access token
    /// for ApiClient and the chat stream.
    pub fn save(&self) {
        self.write_to(&create_storage());
        crate::platform::set_backend_access_token(Some(&self.backend_access_token));
    }

    /// Write [`Settings::to_persisted`] to `storage`, logging a failure.
    fn write_to<S: StorageBackend>(&self, storage: &S) {
        if let Err(e) = storage.set(SETTINGS_KEY, &self.to_persisted()) {
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
        self.api_key_for(self.provider)
    }

    /// API key stored for the given provider (empty for Ollama / None)
    pub fn api_key_for(&self, provider: Provider) -> &str {
        match provider {
            Provider::None | Provider::Ollama => "",
            Provider::Anthropic => &self.anthropic_api_key,
            Provider::Openai => &self.openai_api_key,
            Provider::Openrouter => &self.openrouter_api_key,
        }
    }

    /// Set current API key based on provider.
    ///
    /// A changed value becomes user-provided (or unset when blank) and is
    /// persisted on save. Writing back the unchanged value keeps its source,
    /// so saving the settings modal does not persist an environment key.
    pub fn set_current_api_key(&mut self, key: &str) {
        let provider = self.provider;
        if let Some((current, source)) = self.key_slot(provider)
            && current.as_str() != key
        {
            *current = key.to_string();
            *source = if key.trim().is_empty() {
                KeySource::Unset
            } else {
                KeySource::User
            };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn env_sourced_keys_are_never_persisted() {
        let mut s = Settings::default();
        s.populate_from_env_with(env(&[
            ("ANTHROPIC_API_KEY", "sk-ant-from-env"),
            ("OPENAI_API_KEY", "sk-oai-from-env"),
        ]));
        assert_eq!(s.anthropic_api_key, "sk-ant-from-env");
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::Environment);
        let json = serde_json::to_string(&s.to_persisted()).unwrap();
        assert!(
            !json.contains("sk-ant-from-env") && !json.contains("sk-oai-from-env"),
            "{json}"
        );
        assert!(!json.contains("anthropic_api_key"), "{json}");
    }

    #[test]
    fn typed_key_is_persisted_and_overrides_env() {
        let mut s = Settings::default();
        s.populate_from_env_with(env(&[("ANTHROPIC_API_KEY", "sk-ant-from-env")]));
        s.provider = Provider::Anthropic;
        s.set_current_api_key("sk-ant-typed");
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::User);
        assert_eq!(
            serde_json::to_value(s.to_persisted()).unwrap()["anthropic_api_key"],
            "sk-ant-typed"
        );
    }

    #[test]
    fn saving_the_modal_unchanged_keeps_env_key_unpersisted() {
        let mut s = Settings::default();
        s.populate_from_env_with(env(&[("OPENROUTER_API_KEY", "sk-or-env")]));
        s.provider = Provider::Openrouter;
        s.set_current_api_key("sk-or-env"); // the modal writes back the value it displayed
        assert_eq!(s.key_source(Provider::Openrouter), KeySource::Environment);
        assert!(
            !serde_json::to_string(&s.to_persisted())
                .unwrap()
                .contains("sk-or-env")
        );
    }

    #[test]
    fn key_leaked_by_older_version_is_dropped_when_env_matches() {
        let mut v = serde_json::to_value(Settings::default()).unwrap();
        v["anthropic_api_key"] = "sk-ant-from-env".into();
        let mut s: Settings = serde_json::from_value(v).unwrap();
        s.mark_stored_keys();
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::User);
        s.populate_from_env_with(env(&[("ANTHROPIC_API_KEY", "sk-ant-from-env")]));
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::Environment);
        assert!(
            !serde_json::to_string(&s.to_persisted())
                .unwrap()
                .contains("sk-ant-from-env")
        );
    }

    #[test]
    fn populate_reports_only_a_reclassified_stored_key() {
        let leaked_env = env(&[("ANTHROPIC_API_KEY", "sk-ant-from-env")]);

        // Old-version leak: the stored key equals the environment value.
        let mut s = Settings {
            anthropic_api_key: "sk-ant-from-env".into(),
            ..Settings::default()
        };
        s.mark_stored_keys();
        assert!(s.populate_from_env_with(&leaked_env));

        // No change on disk needed: an empty key filled from the
        // environment, a typed key that differs, no environment at all, and
        // a second pass over already reclassified settings.
        assert!(!Settings::default().populate_from_env_with(&leaked_env));
        let mut typed = Settings {
            anthropic_api_key: "sk-ant-typed".into(),
            ..Settings::default()
        };
        typed.mark_stored_keys();
        assert!(!typed.populate_from_env_with(&leaked_env));
        let mut stored = s.clone();
        assert!(!stored.populate_from_env_with(&leaked_env));
        let mut no_env = Settings {
            openai_api_key: "sk-oai-typed".into(),
            ..Settings::default()
        };
        no_env.mark_stored_keys();
        assert!(!no_env.populate_from_env_with(env(&[])));
    }

    /// Storage held in memory, standing in for the settings file.
    #[derive(Default)]
    struct MemoryStorage(std::cell::RefCell<std::collections::HashMap<String, String>>);

    impl StorageBackend for MemoryStorage {
        fn get<T: serde::de::DeserializeOwned>(
            &self,
            key: &str,
        ) -> Result<T, crate::platform::StorageError> {
            let map = self.0.borrow();
            let raw = map
                .get(key)
                .ok_or_else(|| crate::platform::StorageError(format!("no {key}")))?;
            serde_json::from_str(raw).map_err(|e| crate::platform::StorageError(e.to_string()))
        }

        fn set<T: Serialize>(
            &self,
            key: &str,
            value: &T,
        ) -> Result<(), crate::platform::StorageError> {
            let raw = serde_json::to_string(value)
                .map_err(|e| crate::platform::StorageError(e.to_string()))?;
            self.0.borrow_mut().insert(key.to_string(), raw);
            Ok(())
        }

        fn remove(&self, key: &str) -> Result<(), crate::platform::StorageError> {
            self.0.borrow_mut().remove(key);
            Ok(())
        }

        fn contains_key(&self, key: &str) -> bool {
            self.0.borrow().contains_key(key)
        }
    }

    #[test]
    fn load_rewrites_settings_that_hold_a_leaked_env_key() {
        let storage = MemoryStorage::default();
        let mut leaked = serde_json::to_value(Settings::default()).unwrap();
        leaked["anthropic_api_key"] = "sk-ant-from-env".into();
        leaked["openai_api_key"] = "sk-oai-typed".into();
        storage.set(SETTINGS_KEY, &leaked).unwrap();

        let s = Settings::load_from(
            &storage,
            Some(env(&[("ANTHROPIC_API_KEY", "sk-ant-from-env")])),
        );
        assert_eq!(
            s.anthropic_api_key, "sk-ant-from-env",
            "the key is still used"
        );
        let on_disk = storage.0.borrow()[SETTINGS_KEY].clone();
        assert!(!on_disk.contains("sk-ant-from-env"), "{on_disk}");
        assert!(on_disk.contains("sk-oai-typed"), "{on_disk}");
    }

    #[test]
    fn load_leaves_storage_alone_when_nothing_leaked() {
        let storage = MemoryStorage::default();
        let mut typed = serde_json::to_value(Settings::default()).unwrap();
        typed["anthropic_api_key"] = "sk-ant-typed".into();
        let before = serde_json::to_string(&typed).unwrap();
        storage
            .0
            .borrow_mut()
            .insert(SETTINGS_KEY.into(), before.clone());

        let s = Settings::load_from(&storage, Some(env(&[("ANTHROPIC_API_KEY", "sk-ant-env")])));
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::User);
        assert_eq!(
            storage.0.borrow()[SETTINGS_KEY],
            before,
            "storage was rewritten"
        );

        // Without an environment (the web build) a stored key is never
        // touched, even if it would match.
        let web = MemoryStorage::default();
        let mut leaked = serde_json::to_value(Settings::default()).unwrap();
        leaked["anthropic_api_key"] = "sk-ant-from-env".into();
        let before = serde_json::to_string(&leaked).unwrap();
        web.0
            .borrow_mut()
            .insert(SETTINGS_KEY.into(), before.clone());
        let none: Option<fn(&str) -> Option<String>> = None;
        let s = Settings::load_from(&web, none);
        assert_eq!(s.key_source(Provider::Anthropic), KeySource::User);
        assert_eq!(web.0.borrow()[SETTINGS_KEY], before);
    }

    #[test]
    fn persisted_settings_round_trip_and_keep_backend_token() {
        let mut s = Settings::default();
        s.backend_access_token = "tok-123".into();
        let json = serde_json::to_string(&s.to_persisted()).unwrap();
        assert!(json.contains("\"backend_access_token\":\"tok-123\""));
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.anthropic_api_key, "");
        assert_eq!(back.backend_access_token, "tok-123");
        assert!(
            !serde_json::to_string(&Settings::default().to_persisted())
                .unwrap()
                .contains("backend_access_token")
        );
    }
}
