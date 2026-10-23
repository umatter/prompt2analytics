//! `p2a-chat` — Terminal REPL for conversational analytics
//!
//! A thin SSE client that connects to a running p2a-mcp HTTP server.
//! All LLM calls, tool execution, and session management happen server-side.
//!
//! ## Recording and scripting
//!
//! - `--record transcript.txt` saves a plain-text transcript (no ANSI codes)
//! - `--script prompts.txt` reads prompts from a file instead of stdin
//!
//! Combined, these make sessions fully reproducible:
//! ```bash
//! p2a-chat --script prompts.txt --record transcript.txt --provider anthropic --model claude-sonnet-4-6
//! ```

use std::io::{BufRead, Write as _};
use std::path::PathBuf;

use clap::Parser;
use colored::Colorize;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Client-side types (mirror server types to avoid depending on p2a-mcp)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_results: Option<Vec<ToolResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_type: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmChatRequest {
    pub session_id: String,
    pub message: String,
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
    #[serde(default)]
    pub history: Option<Vec<Message>>,
    #[serde(default = "default_true")]
    pub interpret: bool,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default = "default_true")]
    pub retrieve_history: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "status")]
    Status { message: String },
    #[serde(rename = "tool_start")]
    ToolStart {
        tool: String,
        arguments: serde_json::Value,
    },
    #[serde(rename = "tool_end")]
    ToolEnd {
        tool: String,
        elapsed_ms: u64,
        #[allow(dead_code)]
        result: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        images: Option<Vec<ImageData>>,
    },
    #[serde(rename = "content")]
    Content { text: String },
    #[serde(rename = "done")]
    Done { message: Message },
    #[serde(rename = "error")]
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiResponse<T> {
    #[allow(dead_code)]
    success: bool,
    data: Option<T>,
    #[allow(dead_code)]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateSessionData {
    session_id: String,
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// Interactive chat with a p2a-mcp server
#[derive(Parser, Debug, Clone)]
pub struct ChatArgs {
    /// Server URL
    #[arg(long, default_value = "http://localhost:8080", env = "P2A_SERVER_URL")]
    pub url: String,

    /// LLM provider (openai, anthropic, ollama)
    #[arg(long, default_value = "ollama", env = "P2A_PROVIDER")]
    pub provider: String,

    /// Model name
    #[arg(long, default_value = "llama3.2", env = "P2A_MODEL")]
    pub model: String,

    /// API key for the provider
    #[arg(long, env = "P2A_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Bearer token for a p2a-mcp server started with --access-token
    #[arg(long, env = "P2A_ACCESS_TOKEN", hide_env_values = true)]
    pub access_token: Option<String>,

    /// Override the LLM API base URL
    #[arg(long, env = "P2A_BASE_URL")]
    pub base_url: Option<String>,

    /// Directory for saved images/charts
    #[arg(long, default_value = ".")]
    pub output_dir: PathBuf,

    /// Save a plain-text transcript to this file
    #[arg(long)]
    pub record: Option<PathBuf>,

    /// Read prompts from a file instead of interactive input (one prompt per line,
    /// or multi-line prompts separated by blank lines)
    #[arg(long)]
    pub script: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Transcript recorder
// ---------------------------------------------------------------------------

struct Recorder {
    file: std::fs::File,
}

impl Recorder {
    fn new(path: &PathBuf) -> anyhow::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self { file })
    }

    fn write_banner(&mut self, base_url: &str, provider: &str, model: &str) -> anyhow::Result<()> {
        writeln!(
            self.file,
            "$ p2a chat --provider {} --model {}",
            provider, model
        )?;
        writeln!(self.file, "p2a chat")?;
        writeln!(
            self.file,
            "Connected to {} | Provider: {} ({})",
            base_url, provider, model
        )?;
        writeln!(self.file)?;
        Ok(())
    }

    fn write_prompt(&mut self, input: &str) -> anyhow::Result<()> {
        writeln!(self.file, "p2a> {}", input)?;
        Ok(())
    }

    fn write_tool_start(&mut self, tool: &str, args_brief: &str) -> anyhow::Result<()> {
        write!(self.file, "  [tool] {} {}", tool, args_brief)?;
        Ok(())
    }

    fn write_tool_end(&mut self, elapsed_ms: u64) -> anyhow::Result<()> {
        writeln!(self.file, "  done {}ms", elapsed_ms)?;
        Ok(())
    }

    fn write_content(&mut self, text: &str) -> anyhow::Result<()> {
        write!(self.file, "{}", text)?;
        Ok(())
    }

    fn write_done(&mut self) -> anyhow::Result<()> {
        writeln!(self.file)?;
        writeln!(self.file)?;
        Ok(())
    }

    fn write_image_saved(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        writeln!(self.file, "  [saved] {}", path.display())?;
        Ok(())
    }

    fn write_error(&mut self, error: &str) -> anyhow::Result<()> {
        writeln!(self.file, "Error: {}", error)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Script reader — reads prompts from a file
// ---------------------------------------------------------------------------

fn read_script(path: &PathBuf) -> anyhow::Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut prompts = Vec::new();
    let mut current = String::new();

    for line in reader.lines() {
        let line = line?;
        // Lines starting with # are comments
        if line.starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            // Blank line separates multi-line prompts
            if !current.trim().is_empty() {
                prompts.push(current.trim().to_string());
                current.clear();
            }
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(line.trim());
        }
    }
    if !current.trim().is_empty() {
        prompts.push(current.trim().to_string());
    }

    Ok(prompts)
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

fn parse_sse_line(line: &str) -> Option<StreamEvent> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    if let Some(data) = line.strip_prefix("data: ") {
        match serde_json::from_str::<StreamEvent>(data) {
            Ok(event) => return Some(event),
            Err(e) => {
                log::warn!("Failed to parse SSE event: {} - data: {}", e, data);
                return None;
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Terminal rendering + recording
// ---------------------------------------------------------------------------

fn render_event(event: &StreamEvent, output_dir: &PathBuf, recorder: &mut Option<Recorder>) {
    match event {
        StreamEvent::Status { message } => {
            eprintln!("  {}", message.dimmed());
        }
        StreamEvent::ToolStart { tool, arguments } => {
            let args_brief = format_args_brief(arguments);
            eprint!(
                "  {} {} {}",
                "[tool]".yellow(),
                tool.yellow().bold(),
                args_brief.dimmed()
            );
            if let Some(rec) = recorder.as_mut() {
                let _ = rec.write_tool_start(tool, &args_brief);
            }
        }
        StreamEvent::ToolEnd { elapsed_ms, .. } => {
            eprintln!("  {} {}ms", "done".green(), elapsed_ms);
            if let Some(rec) = recorder.as_mut() {
                let _ = rec.write_tool_end(*elapsed_ms);
            }
        }
        StreamEvent::ToolResult { images } => {
            if let Some(imgs) = images {
                for (i, img) in imgs.iter().enumerate() {
                    let ext = match img.mime_type.as_str() {
                        "image/png" => "png",
                        "image/svg+xml" => "svg",
                        _ => "bin",
                    };
                    let path = output_dir.join(format!("chart_{}.{}", i, ext));
                    match save_image(&img.data, &path) {
                        Ok(()) => {
                            eprintln!("  {} {}", "Saved:".green(), path.display());
                            if let Some(rec) = recorder.as_mut() {
                                let _ = rec.write_image_saved(&path);
                            }
                        }
                        Err(e) => eprintln!("  {} {}", "Image save error:".red(), e),
                    }
                }
            }
        }
        StreamEvent::Content { text } => {
            print!("{}", text);
            let _ = std::io::stdout().flush();
            if let Some(rec) = recorder.as_mut() {
                let _ = rec.write_content(text);
            }
        }
        StreamEvent::Done { .. } => {
            println!();
            if let Some(rec) = recorder.as_mut() {
                let _ = rec.write_done();
            }
        }
        StreamEvent::Error { error } => {
            eprintln!("\n{} {}", "Error:".red().bold(), error);
            if let Some(rec) = recorder.as_mut() {
                let _ = rec.write_error(error);
            }
        }
    }
}

fn format_args_brief(args: &serde_json::Value) -> String {
    if let Some(obj) = args.as_object() {
        let pairs: Vec<String> = obj
            .iter()
            .take(4)
            .map(|(k, v)| {
                let val = match v {
                    serde_json::Value::String(s) => {
                        if s.len() > 30 {
                            format!("\"{}...\"", &s[..27])
                        } else {
                            format!("\"{}\"", s)
                        }
                    }
                    other => {
                        let s = other.to_string();
                        if s.len() > 30 {
                            format!("{}...", &s[..27])
                        } else {
                            s
                        }
                    }
                };
                format!("{}={}", k, val)
            })
            .collect();
        let suffix = if obj.len() > 4 { ", ..." } else { "" };
        format!("({}{})", pairs.join(", "), suffix)
    } else {
        String::new()
    }
}

fn save_image(base64_data: &str, path: &std::path::Path) -> anyhow::Result<()> {
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, base64_data)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP client and connection errors
// ---------------------------------------------------------------------------

/// The command that starts a local server, shown when it cannot be reached.
const START_SERVER_COMMAND: &str =
    "cargo run -p p2a-mcp --features full -- --transport http --port 8080";

/// Client builder with the backend access token, if any, as a default
/// `Authorization: Bearer <token>` header marked sensitive (so it is never
/// shown in debug output). A blank token means none.
fn client_builder(access_token: Option<&str>) -> anyhow::Result<reqwest::ClientBuilder> {
    let builder = reqwest::Client::builder();
    let Some(token) = access_token.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(builder);
    };
    let mut value =
        reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
            anyhow::anyhow!(
                "--access-token / P2A_ACCESS_TOKEN contains characters that cannot be sent in an HTTP header (visible ASCII only)"
            )
        })?;
    value.set_sensitive(true);
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(reqwest::header::AUTHORIZATION, value);
    Ok(builder.default_headers(headers))
}

/// HTTP client for the p2a-mcp server, sending `access_token` as a bearer
/// token on every request when one is given.
pub fn build_http_client(access_token: Option<&str>) -> anyhow::Result<reqwest::Client> {
    Ok(client_builder(access_token)?.build()?)
}

/// Why the initial connection to the server failed.
#[derive(Debug)]
pub enum ConnectError {
    /// The server could not be reached at all.
    Unreachable(String),
    /// The server answered 401: the token is missing or was rejected.
    Unauthorized { token_sent: bool },
    /// The server answered with another non-2xx status.
    Http { status: u16 },
    /// The response was not the expected session JSON.
    BadResponse(String),
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::Unreachable(e) => write!(f, "could not reach the server: {e}"),
            ConnectError::Unauthorized { token_sent } => {
                f.write_str(unauthorized_message(*token_sent))
            }
            ConnectError::Http { status } => write!(f, "the server answered HTTP {status}"),
            ConnectError::BadResponse(e) => write!(f, "unexpected response from the server: {e}"),
        }
    }
}

impl std::error::Error for ConnectError {}

/// What to tell the user about a 401.
fn unauthorized_message(token_sent: bool) -> &'static str {
    if token_sent {
        "The server rejected the access token. Check --access-token or P2A_ACCESS_TOKEN."
    } else {
        "The server requires an access token. Pass --access-token <token> or set P2A_ACCESS_TOKEN."
    }
}

/// What to print when the session cannot be created.
#[derive(Debug)]
struct ConnectHint {
    /// The first line.
    headline: String,
    /// The lines after it.
    details: Vec<String>,
    /// Whether [`START_SERVER_COMMAND`] follows the details, indented.
    start_command: bool,
}

impl ConnectHint {
    /// The hint as plain lines, the start command included.
    fn lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.details.len() + 2);
        lines.push(self.headline.clone());
        lines.extend(self.details.iter().cloned());
        if self.start_command {
            lines.push(format!("  {START_SERVER_COMMAND}"));
        }
        lines
    }
}

/// The hint printed when the session cannot be created. Only an
/// unreachable server gets the "start the server" advice.
fn connect_failure_hint(err: &ConnectError) -> ConnectHint {
    match err {
        ConnectError::Unreachable(_) => ConnectHint {
            headline: "Could not connect to p2a-mcp server.".to_string(),
            details: vec![
                format!("  {err}"),
                String::new(),
                "Start the server first:".to_string(),
            ],
            start_command: true,
        },
        ConnectError::Unauthorized { token_sent } => ConnectHint {
            headline: "The p2a-mcp server refused the request (HTTP 401).".to_string(),
            details: vec![format!("  {}", unauthorized_message(*token_sent))],
            start_command: false,
        },
        ConnectError::Http { .. } | ConnectError::BadResponse(_) => ConnectHint {
            headline: err.to_string(),
            details: Vec::new(),
            start_command: false,
        },
    }
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

pub async fn run(args: &ChatArgs) -> anyhow::Result<()> {
    let token = args
        .access_token
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty());
    let client = build_http_client(token)?;
    let token_sent = token.is_some();
    let base_url = args.url.trim_end_matches('/');

    // Create session
    let session_id = match create_session(&client, base_url, token_sent).await {
        Ok(id) => id,
        Err(err) => {
            let hint = connect_failure_hint(&err);
            eprintln!("{}", hint.headline.red().bold());
            for line in &hint.details {
                eprintln!("{line}");
            }
            if hint.start_command {
                eprintln!("  {}", START_SERVER_COMMAND.bright_cyan());
            }
            return Err(anyhow::Error::new(err));
        }
    };

    // Initialize recorder
    let mut recorder = if let Some(ref path) = args.record {
        let mut rec = Recorder::new(path)?;
        rec.write_banner(base_url, &args.provider, &args.model)?;
        Some(rec)
    } else {
        None
    };

    // Load script prompts if provided
    let script_prompts = if let Some(ref path) = args.script {
        let prompts = read_script(path)?;
        if prompts.is_empty() {
            anyhow::bail!("Script file is empty: {}", path.display());
        }
        eprintln!(
            "{}",
            format!("Loaded {} prompts from {}", prompts.len(), path.display()).dimmed()
        );
        Some(prompts)
    } else {
        None
    };

    // Welcome banner
    println!("{}", "p2a chat".bold());
    println!(
        "Connected to {} | Provider: {} ({}) | Session: {}",
        base_url.bright_cyan(),
        args.provider.bright_cyan(),
        args.model.bright_cyan(),
        &session_id[..8.min(session_id.len())],
    );
    if script_prompts.is_none() {
        println!(
            "Type {} for commands, {} to exit.",
            "/help".yellow(),
            "Ctrl+D".yellow()
        );
    }
    println!();

    let provider_config = ProviderConfig {
        provider_type: args.provider.clone(),
        api_key: args.api_key.clone(),
        base_url: args.base_url.clone(),
        model: args.model.clone(),
        temperature: None,
        max_tokens: None,
    };

    let mut history: Vec<Message> = Vec::new();

    // Choose input source: script file or interactive readline
    if let Some(prompts) = script_prompts {
        // Non-interactive: run each prompt sequentially
        for input in &prompts {
            // Echo the prompt (looks like interactive)
            println!("{} {}", "p2a>".bright_green().bold(), input);
            println!();

            if let Some(ref mut rec) = recorder {
                rec.write_prompt(input)?;
            }

            let request = LlmChatRequest {
                session_id: session_id.clone(),
                message: input.clone(),
                provider: Some(provider_config.clone()),
                history: Some(history.clone()),
                interpret: true,
                conversation_id: None,
                retrieve_history: false,
            };

            match stream_chat(
                &client,
                base_url,
                &request,
                &args.output_dir,
                &mut recorder,
                token_sent,
            )
            .await
            {
                Ok(msg) => {
                    history.push(Message {
                        role: "user".to_string(),
                        content: input.clone(),
                        tool_calls: None,
                        tool_results: None,
                    });
                    history.push(msg);
                }
                Err(e) => {
                    eprintln!("{} {}", "Request failed:".red(), e);
                }
            }
        }
    } else {
        // Interactive mode
        let mut rl = rustyline::DefaultEditor::new()?;

        loop {
            let line = match rl.readline("p2a> ") {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    println!("{}", "Use Ctrl+D or /quit to exit.".dimmed());
                    continue;
                }
                Err(rustyline::error::ReadlineError::Eof) => {
                    println!("{}", "Goodbye.".dimmed());
                    break;
                }
                Err(e) => return Err(e.into()),
            };

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            let _ = rl.add_history_entry(input);

            // Slash commands
            match input {
                "/quit" | "/exit" => {
                    println!("{}", "Goodbye.".dimmed());
                    break;
                }
                "/clear" => {
                    history.clear();
                    println!("{}", "History cleared.".dimmed());
                    continue;
                }
                "/help" => {
                    print_help();
                    continue;
                }
                _ => {}
            }

            if let Some(ref mut rec) = recorder {
                rec.write_prompt(input)?;
            }

            let request = LlmChatRequest {
                session_id: session_id.clone(),
                message: input.to_string(),
                provider: Some(provider_config.clone()),
                history: Some(history.clone()),
                interpret: true,
                conversation_id: None,
                retrieve_history: false,
            };

            match stream_chat(
                &client,
                base_url,
                &request,
                &args.output_dir,
                &mut recorder,
                token_sent,
            )
            .await
            {
                Ok(msg) => {
                    history.push(Message {
                        role: "user".to_string(),
                        content: input.to_string(),
                        tool_calls: None,
                        tool_results: None,
                    });
                    history.push(msg);
                }
                Err(e) => {
                    eprintln!("{} {}", "Request failed:".red(), e);
                }
            }
        }
    }

    Ok(())
}

/// Create a session. `token_sent` only shapes the 401 message; the token
/// itself travels as the client's default header.
async fn create_session(
    client: &reqwest::Client,
    base_url: &str,
    token_sent: bool,
) -> Result<String, ConnectError> {
    let resp = client
        .post(format!("{}/api/sessions", base_url))
        .json(&serde_json::json!({}))
        .send()
        .await
        .map_err(|e| ConnectError::Unreachable(e.to_string()))?;

    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(ConnectError::Unauthorized { token_sent });
    }
    if !status.is_success() {
        return Err(ConnectError::Http {
            status: status.as_u16(),
        });
    }

    let body: ApiResponse<CreateSessionData> = resp
        .json()
        .await
        .map_err(|e| ConnectError::BadResponse(e.to_string()))?;
    let data = body
        .data
        .ok_or_else(|| ConnectError::BadResponse("no data in session response".to_string()))?;
    Ok(data.session_id)
}

async fn stream_chat(
    client: &reqwest::Client,
    base_url: &str,
    request: &LlmChatRequest,
    output_dir: &PathBuf,
    recorder: &mut Option<Recorder>,
    token_sent: bool,
) -> anyhow::Result<Message> {
    let resp = client
        .post(format!("{}/api/llm/chat/stream", base_url))
        .json(request)
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        anyhow::bail!("{}", unauthorized_message(token_sent));
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {}", status, body);
    }

    let mut stream = resp.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut final_message: Option<Message> = None;

    while let Some(chunk_result) = stream.next().await {
        let bytes = chunk_result?;
        // Buffer raw bytes and decode only complete lines, so a multi-byte
        // UTF-8 character split across network chunks is never corrupted by
        // per-chunk lossy decoding.
        buffer.extend_from_slice(&bytes);
        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
            let mut end = line_bytes.len() - 1;
            if end > 0 && line_bytes[end - 1] == b'\r' {
                end -= 1;
            }
            let line = String::from_utf8_lossy(&line_bytes[..end]).into_owned();
            if let Some(event) = parse_sse_line(&line) {
                if let StreamEvent::Done { ref message } = event {
                    final_message = Some(message.clone());
                }
                render_event(&event, output_dir, recorder);
            }
        }
    }

    // Process any trailing bytes with no final newline.
    if !buffer.is_empty() {
        let line = String::from_utf8_lossy(&buffer).into_owned();
        if let Some(event) = parse_sse_line(&line) {
            if let StreamEvent::Done { ref message } = event {
                final_message = Some(message.clone());
            }
            render_event(&event, output_dir, recorder);
        }
    }

    final_message.ok_or_else(|| anyhow::anyhow!("Stream ended without a Done event"))
}

fn print_help() {
    println!("{}", "Commands:".bold());
    println!("  {}    Clear conversation history", "/clear".yellow());
    println!("  {}     Exit the chat", "/quit".yellow());
    println!("  {}     Show this help", "/help".yellow());
    println!();
    println!("{}", "Tips:".bold());
    println!("  - Ask natural language questions about your data");
    println!("  - Reference datasets by filename (e.g., \"load sales.csv\")");
    println!("  - Follow-up questions use conversation context");
    println!("  - Use Ctrl+D to exit, Ctrl+C to cancel");
    println!();
    println!("{}", "Recording:".bold());
    println!("  --record file.txt   Save transcript to file");
    println!("  --script file.txt   Read prompts from file");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// Serve exactly one HTTP request on a loopback port with a canned
    /// response; the raw request text is sent back on the channel.
    fn one_shot_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut raw = Vec::new();
            let mut buf = [0u8; 1024];
            let header_end = loop {
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break None;
                }
                raw.extend_from_slice(&buf[..n]);
                if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    break Some(pos + 4);
                }
            };
            if let Some(end) = header_end {
                let head = String::from_utf8_lossy(&raw[..end]).to_ascii_lowercase();
                let content_length = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while raw.len() < end + content_length {
                    let n = stream.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                }
            }
            let response = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = tx.send(String::from_utf8_lossy(&raw).into_owned());
        });
        (format!("http://{addr}"), rx)
    }

    const OK_SESSION: &str = r#"{"success":true,"data":{"session_id":"abc123"},"error":null}"#;

    #[tokio::test]
    async fn create_session_sends_bearer_token() {
        let (url, rx) = one_shot_server("200 OK", OK_SESSION);
        let client = client_builder(Some("s3cret-token"))
            .unwrap()
            .no_proxy()
            .build()
            .unwrap();
        assert_eq!(create_session(&client, &url, true).await.unwrap(), "abc123");
        let req = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap()
            .to_ascii_lowercase();
        assert!(req.contains("authorization: bearer s3cret-token"), "{req}");
    }

    #[tokio::test]
    async fn no_token_sends_no_authorization_header() {
        let (url, rx) = one_shot_server("200 OK", OK_SESSION);
        let client = client_builder(None).unwrap().no_proxy().build().unwrap();
        create_session(&client, &url, false).await.unwrap();
        assert!(
            !rx.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .to_ascii_lowercase()
                .contains("authorization:")
        );
    }

    #[tokio::test]
    async fn http_401_without_token_asks_for_one() {
        let (url, _rx) = one_shot_server("401 Unauthorized", "missing or invalid bearer token");
        let client = client_builder(None).unwrap().no_proxy().build().unwrap();
        let err = create_session(&client, &url, false).await.unwrap_err();
        assert!(
            matches!(err, ConnectError::Unauthorized { token_sent: false }),
            "{err:?}"
        );
        let hint = connect_failure_hint(&err);
        assert!(!hint.start_command, "{hint:?}");
        let hint = hint.lines().join("\n");
        assert!(
            hint.contains("--access-token") && hint.contains("P2A_ACCESS_TOKEN"),
            "{hint}"
        );
        assert!(!hint.contains("Start the server first"), "{hint}");
    }

    #[tokio::test]
    async fn http_401_with_token_says_it_was_rejected() {
        let (url, _rx) = one_shot_server("401 Unauthorized", "missing or invalid bearer token");
        let client = client_builder(Some("wrong"))
            .unwrap()
            .no_proxy()
            .build()
            .unwrap();
        let err = create_session(&client, &url, true).await.unwrap_err();
        assert!(matches!(
            err,
            ConnectError::Unauthorized { token_sent: true }
        ));
        let hint = connect_failure_hint(&err);
        assert!(!hint.start_command, "{hint:?}");
        assert!(hint.lines().join("\n").contains("rejected"));
    }

    #[tokio::test]
    async fn refused_connection_still_suggests_starting_the_server() {
        // Bind and drop: the port is closed by the time the client connects.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let client = client_builder(None).unwrap().no_proxy().build().unwrap();
        let err = create_session(&client, &format!("http://127.0.0.1:{port}"), false)
            .await
            .unwrap_err();
        assert!(matches!(err, ConnectError::Unreachable(_)), "{err:?}");
        let hint = connect_failure_hint(&err);
        assert!(hint.start_command, "{hint:?}");
        let lines = hint.lines();
        assert!(lines.join("\n").contains("Start the server first"));
        assert_eq!(
            lines.last().map(String::as_str),
            Some(format!("  {START_SERVER_COMMAND}").as_str())
        );
        // The command is printed (highlighted) by run() from the flag, so
        // the details never carry it as text.
        assert!(
            !hint
                .details
                .iter()
                .any(|l| l.contains(START_SERVER_COMMAND))
        );
    }

    #[test]
    fn other_failures_get_a_single_line_and_no_start_command() {
        for err in [
            ConnectError::Http { status: 503 },
            ConnectError::BadResponse("missing session_id".into()),
        ] {
            let hint = connect_failure_hint(&err);
            assert!(!hint.start_command, "{hint:?}");
            assert_eq!(hint.lines(), vec![err.to_string()]);
        }
    }

    #[test]
    fn token_flag_parses_and_secret_env_values_are_hidden() {
        use clap::CommandFactory;
        let cmd = ChatArgs::command();
        for id in ["access_token", "api_key"] {
            let arg = cmd.get_arguments().find(|a| a.get_id() == id).unwrap();
            assert!(arg.is_hide_env_values_set(), "{id}");
        }
        let args = ChatArgs::try_parse_from(["p2a-chat", "--access-token", "t0k"]).unwrap();
        assert_eq!(args.access_token.as_deref(), Some("t0k"));
    }

    #[test]
    fn token_with_control_characters_is_refused_before_sending() {
        let err = build_http_client(Some("bad\r\ntoken"))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "--access-token / P2A_ACCESS_TOKEN contains characters that cannot be sent in an HTTP header (visible ASCII only)",
            "the message names the source and never the value"
        );
    }
}
