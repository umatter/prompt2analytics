//! Session state management

use crate::api::api;

/// Information about a loaded dataset
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

/// Session state for managing the backend session
#[derive(Clone)]
pub struct SessionState {
    /// Current session ID
    pub session_id: Option<String>,
    /// Whether the session has been initialized
    pub is_initialized: bool,
    /// Whether initialization is in progress
    pub is_loading: bool,
    /// Error message if initialization failed
    pub error: Option<String>,
    /// Currently loaded datasets
    pub loaded_datasets: Vec<DatasetInfo>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    /// Create a new session state
    pub fn new() -> Self {
        Self {
            session_id: None,
            is_initialized: false,
            is_loading: false,
            error: None,
            loaded_datasets: Vec::new(),
        }
    }

    /// Initialize the session by creating a new one with the backend
    pub async fn initialize(&self) -> Result<String, String> {
        let client = api();

        match client.create_session().await {
            Ok(session_id) => {
                tracing::info!("Session created: {}", session_id);
                Ok(session_id)
            }
            Err(e) => {
                tracing::error!("Failed to create session: {}", e);
                Err(e)
            }
        }
    }

    /// Ensure we have a valid session, creating one if necessary
    pub async fn ensure_session(&mut self) -> Result<String, String> {
        if let Some(ref id) = self.session_id {
            // Verify session is still valid
            let client = api();
            match client.get_session(id).await {
                Ok(_) => return Ok(id.clone()),
                Err(_) => {
                    // Session expired, create new one
                    tracing::info!("Session expired, creating new one");
                }
            }
        }

        // Create new session
        let session_id = self.initialize().await?;
        self.session_id = Some(session_id.clone());
        self.is_initialized = true;
        Ok(session_id)
    }

    /// Get the current session ID if available
    pub fn get_session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Check if session is ready
    pub fn is_ready(&self) -> bool {
        self.is_initialized && self.session_id.is_some()
    }

    /// Set the session ID
    pub fn set_session_id(&mut self, id: String) {
        self.session_id = Some(id);
        self.is_initialized = true;
        self.error = None;
    }

    /// Set error state
    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.is_loading = false;
    }

    /// Clear error state
    pub fn clear_error(&mut self) {
        self.error = None;
    }

    /// Set loading state
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }

    /// Add a loaded dataset
    pub fn add_dataset(&mut self, name: String, rows: usize, cols: usize) {
        // Remove if already exists
        self.loaded_datasets.retain(|d| d.name != name);
        self.loaded_datasets.push(DatasetInfo { name, rows, cols });
    }

    /// Get loaded datasets
    pub fn get_datasets(&self) -> &[DatasetInfo] {
        &self.loaded_datasets
    }

    /// Clear all datasets
    pub fn clear_datasets(&mut self) {
        self.loaded_datasets.clear();
    }
}
