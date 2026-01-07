//! File-related Tauri commands.

use tauri_plugin_dialog::DialogExt;

/// File filter for data files.
#[derive(Debug, Clone)]
pub struct DataFileFilter {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}

/// Default filters for data files.
pub const DATA_FILE_FILTERS: &[DataFileFilter] = &[
    DataFileFilter {
        name: "CSV Files",
        extensions: &["csv"],
    },
    DataFileFilter {
        name: "Parquet Files",
        extensions: &["parquet"],
    },
    DataFileFilter {
        name: "Excel Files",
        extensions: &["xlsx", "xls", "xlsb"],
    },
    DataFileFilter {
        name: "Stata Files",
        extensions: &["dta"],
    },
    DataFileFilter {
        name: "SAS Files",
        extensions: &["sas7bdat"],
    },
    DataFileFilter {
        name: "All Data Files",
        extensions: &["csv", "parquet", "xlsx", "xls", "dta", "sas7bdat"],
    },
];

/// Open a file dialog to pick a data file.
#[tauri::command]
pub async fn pick_file(window: tauri::Window) -> Result<Option<String>, String> {
    let file_path = window
        .dialog()
        .file()
        .add_filter("Data Files", &["csv", "parquet", "xlsx", "xls", "dta", "sas7bdat"])
        .add_filter("CSV", &["csv"])
        .add_filter("Parquet", &["parquet"])
        .add_filter("Excel", &["xlsx", "xls"])
        .add_filter("All Files", &["*"])
        .blocking_pick_file();

    Ok(file_path.map(|p| p.to_string()))
}

/// Open a file dialog to pick multiple data files.
#[tauri::command]
pub async fn pick_files(window: tauri::Window) -> Result<Vec<String>, String> {
    let file_paths = window
        .dialog()
        .file()
        .add_filter("Data Files", &["csv", "parquet", "xlsx", "xls", "dta", "sas7bdat"])
        .add_filter("All Files", &["*"])
        .blocking_pick_files();

    Ok(file_paths
        .map(|paths| paths.into_iter().map(|p| p.to_string()).collect())
        .unwrap_or_default())
}

/// Open a directory picker dialog.
#[tauri::command]
pub async fn pick_directory(window: tauri::Window) -> Result<Option<String>, String> {
    let dir_path = window.dialog().file().blocking_pick_folder();

    Ok(dir_path.map(|p| p.to_string()))
}
