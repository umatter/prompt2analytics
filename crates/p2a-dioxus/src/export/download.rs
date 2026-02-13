//! Platform-specific download and clipboard utilities

use super::types::ExportFormat;

/// Trigger a file download with the given content
///
/// On web: Creates a Blob URL and triggers an anchor click
/// On desktop: Opens a save file dialog
pub fn trigger_download(content: &str, filename: &str, format: ExportFormat) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        trigger_download_web(content, filename, format)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        trigger_download_native(content, filename, format)
    }
}

/// Copy content to clipboard
///
/// On web: Uses navigator.clipboard API
/// On desktop: Uses system clipboard
pub fn copy_to_clipboard(content: &str) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        copy_to_clipboard_web(content)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        copy_to_clipboard_native(content)
    }
}

// ============================================================================
// Web (WASM) implementation
// ============================================================================

#[cfg(target_arch = "wasm32")]
fn trigger_download_web(content: &str, filename: &str, format: ExportFormat) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("No window object")?;
    let document = window.document().ok_or("No document object")?;

    // Create blob with content
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&wasm_bindgen::JsValue::from_str(content));

    let mut blob_options = web_sys::BlobPropertyBag::new();
    blob_options.type_(format.mime_type());

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)
        .map_err(|e| format!("Failed to create blob: {:?}", e))?;

    // Create object URL
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("Failed to create object URL: {:?}", e))?;

    // Create and click anchor element
    let anchor = document
        .create_element("a")
        .map_err(|e| format!("Failed to create anchor: {:?}", e))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "Failed to cast to anchor")?;

    anchor.set_href(&url);
    anchor.set_download(filename);

    // Append to body, click, and remove
    let body = document.body().ok_or("No body element")?;
    body.append_child(&anchor)
        .map_err(|e| format!("Failed to append anchor: {:?}", e))?;

    anchor.click();

    body.remove_child(&anchor)
        .map_err(|e| format!("Failed to remove anchor: {:?}", e))?;

    // Revoke URL after a short delay to ensure download starts
    // Note: In a real app we might want to use a timeout, but for simplicity we revoke immediately
    let _ = web_sys::Url::revoke_object_url(&url);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn copy_to_clipboard_web(content: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or("No window object")?;
    let navigator = window.navigator();
    let clipboard = navigator.clipboard();

    // Note: This returns a Promise, but we don't await it here
    // The clipboard write happens asynchronously
    let content_js = wasm_bindgen::JsValue::from_str(content);
    let _ = clipboard.write_text(&content_js.as_string().unwrap_or_default());

    Ok(())
}

// ============================================================================
// Native (Desktop) implementation
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
fn trigger_download_native(
    content: &str,
    filename: &str,
    _format: ExportFormat,
) -> Result<(), String> {
    // For native, we just write to the current directory
    // A full implementation would use rfd for file dialogs
    // Strip directory components to prevent path traversal attacks
    let safe_name = std::path::Path::new(filename)
        .file_name()
        .ok_or_else(|| "Invalid filename: no file component found".to_string())?;
    let path = std::path::Path::new(safe_name);
    std::fs::write(path, content).map_err(|e| format!("Failed to write file: {}", e))?;
    tracing::info!("Exported to: {}", path.display());
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn copy_to_clipboard_native(content: &str) -> Result<(), String> {
    // For now, just log - a full implementation would use arboard crate
    tracing::info!("Copy to clipboard (native): {} bytes", content.len());

    // Try to use xclip/xsel on Linux, pbcopy on macOS, or clip on Windows
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Try xclip first, then xsel
        if let Ok(mut child) = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(content.as_bytes());
            }
            let _ = child.wait();
            return Ok(());
        }

        if let Ok(mut child) = Command::new("xsel")
            .args(["--clipboard", "--input"])
            .stdin(Stdio::piped())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(content.as_bytes());
            }
            let _ = child.wait();
            return Ok(());
        }

        Err("No clipboard utility found (install xclip or xsel)".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("pbcopy")
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to run pbcopy: {}", e))?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(content.as_bytes())
                .map_err(|e| format!("Failed to write to pbcopy: {}", e))?;
        }

        child
            .wait()
            .map_err(|e| format!("Failed to wait for pbcopy: {}", e))?;

        Ok(())
    }

    #[cfg(target_os = "windows")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("clip")
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to run clip: {}", e))?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(content.as_bytes())
                .map_err(|e| format!("Failed to write to clip: {}", e))?;
        }

        child
            .wait()
            .map_err(|e| format!("Failed to wait for clip: {}", e))?;

        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err("Clipboard not supported on this platform".to_string())
    }
}

/// Generate a filename for the export based on conversation title and format
pub fn generate_filename(title: Option<&str>, format: ExportFormat) -> String {
    let base = title
        .map(sanitize_filename)
        .unwrap_or_else(|| "conversation".to_string());

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

    format!("{}_{}.{}", base, timestamp, format.extension())
}

/// Sanitize a string for use as a filename
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' => c,
            _ => '_',
        })
        .collect::<String>()
        .trim()
        .replace("  ", " ")
        .replace(' ', "_")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_filename() {
        let filename = generate_filename(Some("My Analysis"), ExportFormat::Json);
        assert!(filename.starts_with("my_analysis_"));
        assert!(filename.ends_with(".json"));
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Hello World"), "hello_world");
        assert_eq!(sanitize_filename("test/file:name"), "test_file_name");
        assert_eq!(sanitize_filename("Analysis <Report>"), "analysis__report_");
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Markdown.extension(), "md");
        assert_eq!(ExportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_format_mime_type() {
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
        assert_eq!(ExportFormat::Markdown.mime_type(), "text/markdown");
        assert_eq!(ExportFormat::Html.mime_type(), "text/html");
    }
}
