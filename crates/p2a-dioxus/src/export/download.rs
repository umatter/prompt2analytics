//! Platform-specific download and clipboard utilities

use super::types::ExportFormat;

/// Trigger a file download with the given content.
///
/// On web: creates a Blob URL and clicks an anchor; the browser saves the
/// file (`Ok(None)`). On desktop: writes the file into the downloads folder
/// (the home folder if there is none) under a name not yet taken, and
/// returns its full path (`Ok(Some(path))`); there is no save dialog.
pub fn trigger_download(
    content: &str,
    filename: &str,
    format: ExportFormat,
) -> Result<Option<String>, String> {
    download_bytes(content.as_bytes(), filename, format.mime_type())
}

/// Downloads `bytes` as `filename` with MIME type `mime`.
///
/// On web the browser downloads the file (`Ok(None)`); on desktop it is
/// written to the downloads folder and its path is returned.
pub fn download_bytes(bytes: &[u8], filename: &str, mime: &str) -> Result<Option<String>, String> {
    #[cfg(target_arch = "wasm32")]
    {
        trigger_download_web(bytes, filename, mime).map(|()| None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime;
        trigger_download_native(bytes, filename).map(Some)
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
fn trigger_download_web(bytes: &[u8], filename: &str, mime: &str) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("No window object")?;
    let document = window.document().ok_or("No document object")?;

    // Create blob with content
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&js_sys::Uint8Array::from(bytes));

    let mut blob_options = web_sys::BlobPropertyBag::new();
    blob_options.type_(mime);

    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &blob_options)
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

/// Where a desktop export of `filename` is written: in the downloads folder,
/// else the home folder (never the working directory), under the file name
/// alone (directory parts are dropped); a name `exists` reports as taken
/// gets ` (2)`, ` (3)`, ... before its extension.
#[cfg(not(target_arch = "wasm32"))]
fn export_target(
    download_dir: Option<std::path::PathBuf>,
    home_dir: Option<std::path::PathBuf>,
    filename: &str,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Result<std::path::PathBuf, String> {
    use std::path::Path;
    let dir = download_dir
        .or(home_dir)
        .ok_or_else(|| "Cannot export: no downloads or home folder was found".to_string())?;
    let name = Path::new(filename)
        .file_name()
        .ok_or_else(|| "Invalid filename: no file component found".to_string())?;
    let first = dir.join(name);
    if !exists(&first) {
        return Ok(first);
    }
    let name = Path::new(name);
    let stem = name
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = name
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    (2..1000)
        .map(|n| dir.join(format!("{stem} ({n}){ext}")))
        .find(|p| !exists(p))
        .ok_or_else(|| {
            format!(
                "Cannot export: too many files named {stem}{ext} in {}",
                dir.display()
            )
        })
}

#[cfg(not(target_arch = "wasm32"))]
fn trigger_download_native(bytes: &[u8], filename: &str) -> Result<String, String> {
    use std::io::Write;

    #[cfg(any(feature = "desktop", feature = "mobile"))]
    let (downloads, home) = (dirs::download_dir(), dirs::home_dir());
    #[cfg(not(any(feature = "desktop", feature = "mobile")))]
    let (downloads, home) = (None, None);

    let path = export_target(downloads, home, filename, |p| p.exists())?;
    // create_new: a file that appeared since the check is not overwritten.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
    file.write_all(bytes)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
    tracing::info!("Exported to: {}", path.display());
    Ok(path.display().to_string())
}

/// Run `program` with `args`, write `content` to its standard input and
/// close it, and wait: a tool that cannot start or exits with a failure
/// status is an error naming the tool and the status. Its output is
/// discarded (a clipboard tool that stays in the background to serve the
/// selection must not hold a pipe the caller reads).
#[cfg(not(target_arch = "wasm32"))]
fn pipe_to(program: &str, args: &[&str], content: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{program}: {e}"))?;
    // Taken and dropped here, so the tool sees the end of its input.
    let written = child
        .stdin
        .take()
        .map(|mut stdin| stdin.write_all(content.as_bytes()));
    let status = child.wait().map_err(|e| format!("{program}: {e}"))?;
    if !status.success() {
        return Err(format!("{program} failed ({status})"));
    }
    if let Some(Err(e)) = written {
        return Err(format!("{program}: writing failed: {e}"));
    }
    Ok(())
}

/// The Linux clipboard tools tried in order: wl-copy under Wayland, then
/// xclip and xsel.
#[cfg(all(not(target_arch = "wasm32"), any(target_os = "linux", test)))]
fn linux_clipboard_tools(wayland: bool) -> Vec<(&'static str, &'static [&'static str])> {
    let mut tools: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    if wayland {
        tools.push(("wl-copy", &[]));
    }
    tools.push(("xclip", &["-selection", "clipboard"]));
    tools.push(("xsel", &["--clipboard", "--input"]));
    tools
}

#[cfg(not(target_arch = "wasm32"))]
fn copy_to_clipboard_native(content: &str) -> Result<(), String> {
    tracing::info!("Copy to clipboard (native): {} bytes", content.len());

    #[cfg(target_os = "linux")]
    {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let mut tried = Vec::new();
        for (program, args) in linux_clipboard_tools(wayland) {
            match pipe_to(program, args, content) {
                Ok(()) => return Ok(()),
                Err(e) => tried.push(e),
            }
        }
        Err(format!(
            "No clipboard tool worked (install wl-copy, xclip or xsel): {}",
            tried.join("; ")
        ))
    }

    #[cfg(target_os = "macos")]
    {
        pipe_to("pbcopy", &[], content)
    }

    #[cfg(target_os = "windows")]
    {
        pipe_to("clip", &[], content)
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

    #[cfg(not(target_arch = "wasm32"))]
    use std::path::{Path, PathBuf};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn export_target_prefers_downloads_then_home() {
        let d = PathBuf::from("/x/Downloads");
        let h = PathBuf::from("/x/home");
        assert_eq!(
            export_target(Some(d.clone()), Some(h.clone()), "r.html", |_| false),
            Ok(d.join("r.html"))
        );
        assert_eq!(
            export_target(None, Some(h.clone()), "r.html", |_| false),
            Ok(h.join("r.html"))
        );
        let err = export_target(None, None, "r.html", |_| false).unwrap_err();
        assert!(err.contains("no downloads or home folder"), "{err}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn export_target_picks_a_free_name_and_keeps_only_the_file_name() {
        let d = PathBuf::from("/x/Downloads");
        let taken = [d.join("r.html"), d.join("r (2).html")];
        let exists = |p: &Path| taken.iter().any(|t| t == p);
        assert_eq!(
            export_target(Some(d.clone()), None, "r.html", exists),
            Ok(d.join("r (3).html"))
        );
        let only_first = |p: &Path| p == d.join("r.html");
        assert_eq!(
            export_target(Some(d.clone()), None, "r.html", only_first),
            Ok(d.join("r (2).html"))
        );
        assert_eq!(
            export_target(Some(d.clone()), None, "../../etc/r.html", |_| false),
            Ok(d.join("r.html"))
        );
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn pipe_to_reports_the_tools_exit_status() {
        assert_eq!(pipe_to("cat", &[], "x"), Ok(()));
        let err = pipe_to("false", &[], "x").unwrap_err();
        assert!(
            err.contains("false") && err.contains("exit status"),
            "{err}"
        );
        let err = pipe_to("p2a-no-such-clipboard-tool", &[], "x").unwrap_err();
        assert!(err.contains("p2a-no-such-clipboard-tool"), "{err}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn linux_clipboard_tools_in_order() {
        let names = |w| {
            linux_clipboard_tools(w)
                .iter()
                .map(|(p, _)| *p)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(true), ["wl-copy", "xclip", "xsel"]);
        assert_eq!(names(false), ["xclip", "xsel"]);
    }

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
