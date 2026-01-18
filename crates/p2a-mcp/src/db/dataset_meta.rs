//! Dataset metadata CRUD operations

use surrealdb::RecordId;

use super::connection::{DbConnection, DbError};
use super::models::DatasetMeta;

impl DbConnection {
    // ==================== Dataset Meta Operations ====================

    /// Save dataset metadata (upsert based on session_id + name)
    pub async fn save_dataset_meta(&self, meta: &DatasetMeta) -> Result<DatasetMeta, DbError> {
        // Check if a dataset with the same name exists in this session
        let existing = self
            .get_dataset_meta(&meta.session_id, &meta.name)
            .await?;

        if let Some(existing_meta) = existing {
            // Update existing record
            let result: Option<DatasetMeta> = self
                .db()
                .update(("dataset_meta", existing_meta.id_string().as_str()))
                .content(DatasetMeta {
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
                })
                .await?;

            result.ok_or_else(|| DbError::Query("Failed to update dataset meta".to_string()))
        } else {
            // Create new record
            let created: Option<DatasetMeta> = self
                .db()
                .create(meta.id.clone())
                .content(meta.clone())
                .await?;

            created.ok_or_else(|| DbError::Query("Failed to create dataset meta".to_string()))
        }
    }

    /// Get all datasets for a session
    pub async fn get_datasets_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<DatasetMeta>, DbError> {
        let session_id_owned = session_id.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM dataset_meta WHERE session_id = $session_id ORDER BY loaded_at DESC")
            .bind(("session_id", session_id_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;
        Ok(datasets)
    }

    /// Get a specific dataset by session_id and name
    pub async fn get_dataset_meta(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<Option<DatasetMeta>, DbError> {
        let session_id_owned = session_id.to_string();
        let name_owned = name.to_string();
        let mut result = self
            .db()
            .query("SELECT * FROM dataset_meta WHERE session_id = $session_id AND name = $name")
            .bind(("session_id", session_id_owned))
            .bind(("name", name_owned))
            .await?;

        let datasets: Vec<DatasetMeta> = result.take(0)?;
        Ok(datasets.into_iter().next())
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
