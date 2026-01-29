//! Dataset metadata CRUD operations

use tracing;

use super::connection::{DbConnection, DbError};
use super::models::DatasetMeta;

impl DbConnection {
    // ==================== Dataset Meta Operations ====================

    /// Save dataset metadata (upsert based on session_id + name)
    pub async fn save_dataset_meta(&self, meta: &DatasetMeta) -> Result<DatasetMeta, DbError> {
        tracing::info!(
            session_id = %meta.session_id,
            name = %meta.name,
            column_names_to_save = ?meta.column_names,
            column_names_len = meta.column_names.len(),
            "[DB_SAVE_META] Starting save_dataset_meta"
        );

        // Check if a dataset with the same name exists in this session
        let existing = self.get_dataset_meta(&meta.session_id, &meta.name).await?;

        if let Some(existing_meta) = existing {
            tracing::info!(
                existing_id = %existing_meta.id_string(),
                existing_column_names = ?existing_meta.column_names,
                "[DB_SAVE_META] Found existing record, will update"
            );

            // Update existing record
            let update_content = DatasetMeta {
                id: existing_meta.id.clone(),
                session_id: meta.session_id.clone(),
                name: meta.name.clone(),
                source_path: meta.source_path.clone(),
                source_type: meta.source_type.clone(),
                row_count: meta.row_count,
                column_count: meta.column_count,
                column_names: meta.column_names.clone(),
                loaded_at: meta.loaded_at.clone(),
                file_size_bytes: meta.file_size_bytes,
            };

            tracing::info!(
                update_column_names = ?update_content.column_names,
                update_column_names_len = update_content.column_names.len(),
                "[DB_SAVE_META] Content being sent to UPDATE"
            );

            let result: Option<DatasetMeta> = self
                .db()
                .update(("dataset_meta", existing_meta.id_string().as_str()))
                .content(update_content)
                .await?;

            match &result {
                Some(updated) => {
                    tracing::info!(
                        updated_id = %updated.id_string(),
                        updated_column_names = ?updated.column_names,
                        updated_column_names_len = updated.column_names.len(),
                        "[DB_SAVE_META] UPDATE returned successfully"
                    );
                }
                None => {
                    tracing::error!("[DB_SAVE_META] UPDATE returned None!");
                }
            }

            result.ok_or_else(|| DbError::Query("Failed to update dataset meta".to_string()))
        } else {
            tracing::info!(
                new_id = %meta.id_string(),
                "[DB_SAVE_META] No existing record, will create new"
            );

            // Create new record
            let created: Option<DatasetMeta> = self
                .db()
                .create(meta.id.clone())
                .content(meta.clone())
                .await?;

            match &created {
                Some(new_meta) => {
                    tracing::info!(
                        created_id = %new_meta.id_string(),
                        created_column_names = ?new_meta.column_names,
                        created_column_names_len = new_meta.column_names.len(),
                        "[DB_SAVE_META] CREATE returned successfully"
                    );
                }
                None => {
                    tracing::error!("[DB_SAVE_META] CREATE returned None!");
                }
            }

            created.ok_or_else(|| DbError::Query("Failed to create dataset meta".to_string()))
        }
    }

    /// Get all datasets for a session
    pub async fn get_datasets_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<DatasetMeta>, DbError> {
        tracing::info!(
            session_id = %session_id,
            "[DB_GET_DATASETS] Querying datasets for session"
        );

        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query(
                "SELECT * FROM dataset_meta WHERE session_id = $session_id ORDER BY loaded_at DESC",
            )
            .bind(("session_id", session_id_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;

        tracing::info!(
            num_datasets = datasets.len(),
            "[DB_GET_DATASETS] Query returned datasets"
        );

        for (idx, ds) in datasets.iter().enumerate() {
            tracing::info!(
                idx = idx,
                id = %ds.id_string(),
                name = %ds.name,
                column_names = ?ds.column_names,
                column_names_len = ds.column_names.len(),
                "[DB_GET_DATASETS] Dataset record from SurrealDB"
            );
        }

        Ok(datasets)
    }

    /// Get a specific dataset by session_id and name
    pub async fn get_dataset_meta(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<Option<DatasetMeta>, DbError> {
        tracing::debug!(
            session_id = %session_id,
            name = %name,
            "[DB_GET_META] Looking up specific dataset"
        );

        let session_id_owned = session_id.to_string();
        let name_owned = name.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM dataset_meta WHERE session_id = $session_id AND name = $name")
            .bind(("session_id", session_id_owned))
            .bind(("name", name_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;
        let found = datasets.into_iter().next();

        match &found {
            Some(meta) => {
                tracing::debug!(
                    id = %meta.id_string(),
                    column_names = ?meta.column_names,
                    column_names_len = meta.column_names.len(),
                    "[DB_GET_META] Found dataset"
                );
            }
            None => {
                tracing::debug!("[DB_GET_META] Dataset not found");
            }
        }

        Ok(found)
    }

    /// Get a dataset meta by ID
    pub async fn get_dataset_meta_by_id(&self, id: &str) -> Result<DatasetMeta, DbError> {
        let meta: Option<DatasetMeta> = self.db().select(("dataset_meta", id)).await?;
        meta.ok_or_else(|| DbError::NotFound(format!("Dataset meta not found: {}", id)))
    }

    /// Delete dataset metadata by session_id and name
    pub async fn delete_dataset_meta(&self, session_id: &str, name: &str) -> Result<(), DbError> {
        let session_id_owned = session_id.to_string();
        let name_owned = name.to_string();

        self.db()
            .query("DELETE FROM dataset_meta WHERE session_id = $session_id AND name = $name")
            .bind(("session_id", session_id_owned))
            .bind(("name", name_owned))
            .await?;

        Ok(())
    }

    /// Delete all dataset metadata for a session
    pub async fn delete_all_dataset_meta_for_session(
        &self,
        session_id: &str,
    ) -> Result<u32, DbError> {
        let session_id_owned = session_id.to_string();

        // Count before deletion
        let mut result = self
            .db()
            .query("SELECT count() FROM dataset_meta WHERE session_id = $session_id GROUP ALL")
            .bind(("session_id", session_id_owned.clone()))
            .await?;

        #[derive(serde::Deserialize)]
        struct CountResult {
            count: i64,
        }

        let count: Option<CountResult> = result.take(0)?;
        let deleted_count = count.map(|c| c.count as u32).unwrap_or(0);

        // Delete all
        self.db()
            .query("DELETE FROM dataset_meta WHERE session_id = $session_id")
            .bind(("session_id", session_id_owned))
            .await?;

        Ok(deleted_count)
    }

    /// Update the source path for a dataset (useful when file moves)
    pub async fn update_dataset_source_path(
        &self,
        session_id: &str,
        name: &str,
        source_path: &str,
    ) -> Result<DatasetMeta, DbError> {
        let session_id_owned = session_id.to_string();
        let name_owned = name.to_string();

        let mut result = self
            .db()
            .query(
                r#"UPDATE dataset_meta SET
                    source_path = $source_path
                WHERE session_id = $session_id AND name = $name RETURN AFTER"#,
            )
            .bind(("session_id", session_id_owned))
            .bind(("name", name_owned))
            .bind(("source_path", source_path.to_string()))
            .await?;

        let updated: Option<DatasetMeta> = result.take(0)?;
        updated.ok_or_else(|| {
            DbError::NotFound(format!(
                "Dataset meta not found: session={}, name={}",
                session_id, name
            ))
        })
    }
}
