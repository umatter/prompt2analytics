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
        /// The tool that produced the images (absent from older servers)
        #[serde(default)]
        tool: Option<String>,
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

    /// Bearer token for a p2a-mcp server started with --access-token (visible ASCII, no spaces)
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

    /// Read prompts from a file instead of interactive input: prompts are separated by
    /// blank lines and keep their line breaks. Lines starting with # are comments; start a
    /// prompt line with \# to send a #
    #[arg(long)]
    pub script: Option<PathBuf>,

    /// With --script: send the remaining prompts after a failed one; the exit status is
    /// still non-zero
    #[arg(long)]
    pub keep_going: bool,
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

    /// The banner: first the command that reproduces the run (values shell-quoted; the API
    /// key and the access token are never written, a comment line says when they were
    /// given), then the connection line.
    fn write_banner(&mut self, args: &ChatArgs, base_url: &str) -> anyhow::Result<()> {
        let mut cmd = format!(
            "$ p2a chat --url {} --provider {} --model {}",
            quote(&args.url),
            quote(&args.provider),
            quote(&args.model)
        );
        if let Some(b) = &args.base_url {
            cmd.push_str(&format!(" --base-url {}", quote(b)));
        }
        if args.output_dir != std::path::Path::new(".") {
            cmd.push_str(&format!(
                " --output-dir {}",
                quote(&args.output_dir.to_string_lossy())
            ));
        }
        if let Some(s) = &args.script {
            cmd.push_str(&format!(" --script {}", quote(&s.to_string_lossy())));
        }
        if args.keep_going {
            cmd.push_str(" --keep-going");
        }
        if let Some(r) = &args.record {
            cmd.push_str(&format!(" --record {}", quote(&r.to_string_lossy())));
        }
        writeln!(self.file, "{cmd}")?;
        if args.api_key.is_some() || args.access_token.is_some() {
            writeln!(
                self.file,
                "# --api-key / --access-token given, not recorded"
            )?;
        }
        writeln!(self.file, "p2a chat")?;
        writeln!(
            self.file,
            "Connected to {} | Provider: {} ({})",
            base_url, args.provider, args.model
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

    fn write_cancelled(&mut self) -> anyhow::Result<()> {
        writeln!(self.file, "[cancelled]")?;
        writeln!(self.file)?;
        Ok(())
    }
}

/// `s` as one shell word: bare when it is non-empty and only `[A-Za-z0-9_./:=@%+-]`, else in
/// single quotes with each `'` written `'\''` (the rule of p2a-cli's `shell_quote`).
fn quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_./:=@%+-".contains(c))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

// ---------------------------------------------------------------------------
// Script reader — reads prompts from a file
// ---------------------------------------------------------------------------

/// The prompts of a script file. Blank lines separate prompts; a prompt's lines keep their
/// line breaks (each line without trailing white space, the prompt trimmed). A line that
/// starts with `#` is a comment; `\#` at the start of a line stands for a `#`.
fn read_script(path: &PathBuf) -> anyhow::Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut prompts = Vec::new();
    let mut current = String::new();
    let mut flush = |current: &mut String| {
        let prompt = current.trim();
        if !prompt.is_empty() {
            prompts.push(prompt.to_string());
        }
        current.clear();
    };

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut current);
            continue;
        }
        let line = match line.strip_prefix("\\#") {
            Some(rest) => format!("#{rest}"),
            None => line,
        };
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line.trim_end());
    }
    flush(&mut current);

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

fn render_event(
    event: &StreamEvent,
    output_dir: &std::path::Path,
    recorder: &mut Option<Recorder>,
    charts: &mut ChartNamer,
) {
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
        StreamEvent::ToolResult { tool, images } => {
            if let Some(imgs) = images {
                for img in imgs {
                    match charts.save(output_dir, tool.as_deref(), img) {
                        Ok(path) => {
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
                    // Shortened by characters: a byte cut panics inside a multi-byte character.
                    serde_json::Value::String(s) => format!("\"{}\"", p2a_text::ellipsize(s, 30)),
                    other => p2a_text::ellipsize(&other.to_string(), 30).into_owned(),
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

/// The short session id printed in the welcome banner.
fn session_label(id: &str) -> &str {
    p2a_text::truncate_chars(id, 8)
}

/// The file extension of a chart of MIME type `mime` (`bin` for an unknown type).
fn chart_extension(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/svg+xml" => "svg",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "bin",
    }
}

/// A tool name as part of a file name: ASCII letters, digits, `_` and `-` kept, every other
/// character `_`, at most 40 characters.
fn chart_tool_part(tool: &str) -> String {
    let safe: String = tool
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    p2a_text::truncate_chars(&safe, 40).to_string()
}

/// The most file names [`ChartNamer::save`] tries for one chart.
const MAX_CHART_TRIES: u32 = 10_000;

/// Names the charts of one run `chart_<NNN>[_<tool>].<ext>`, numbered from 1 over the
/// whole run, and never overwrites a file: a taken name moves on to the next number.
#[derive(Debug, Default)]
struct ChartNamer {
    /// The last number used
    last: u32,
}

impl ChartNamer {
    /// Decodes `img` and writes it to a new file in `dir`; returns its path.
    fn save(
        &mut self,
        dir: &std::path::Path,
        tool: Option<&str>,
        img: &ImageData,
    ) -> anyhow::Result<PathBuf> {
        use std::io::Write as _;
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &img.data)?;
        let ext = chart_extension(&img.mime_type);
        let tool = tool.map(chart_tool_part).filter(|t| !t.is_empty());
        for _ in 0..MAX_CHART_TRIES {
            self.last += 1;
            let name = match &tool {
                Some(t) => format!("chart_{:03}_{t}.{ext}", self.last),
                None => format!("chart_{:03}.{ext}", self.last),
            };
            let path = dir.join(name);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    file.write_all(&bytes)?;
                    return Ok(path);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        anyhow::bail!(
            "no free chart file name in {} after {MAX_CHART_TRIES} tries",
            dir.display()
        )
    }
}

// ---------------------------------------------------------------------------
// HTTP client and connection errors
// ---------------------------------------------------------------------------

/// The command that starts a local server, shown when it cannot be reached.
const START_SERVER_COMMAND: &str =
    "cargo run -p p2a-mcp --features full -- --transport http --port 8080";

/// The error for an access token that breaks the visible-ASCII rule. It
/// states the rule, in the Dioxus app's wording, and never the value.
const ACCESS_TOKEN_RULE_ERROR: &str = "--access-token / P2A_ACCESS_TOKEN may contain only visible ASCII characters ('!' to '~'), no spaces or tabs";

/// Client builder with the backend access token, if any, as a default
/// `Authorization: Bearer <token>` header marked sensitive (so it is never
/// shown in debug output). A blank token means none.
///
/// The trimmed token may contain only visible ASCII (`'!'..='~'`), so that
/// p2a-chat accepts exactly the tokens the Dioxus app's
/// `normalize_access_token` accepts. `HeaderValue` alone would also accept
/// inner spaces, tabs and non-ASCII bytes. Of those, only non-ASCII bytes
/// fail on the server, which cannot read such a header value at all; inner
/// spaces and tabs it would compare literally with its configured token.
/// They are refused here so that both clients accept the same tokens.
fn client_builder(access_token: Option<&str>) -> anyhow::Result<reqwest::ClientBuilder> {
    let builder = reqwest::Client::builder();
    let Some(token) = access_token.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(builder);
    };
    let unsendable = || anyhow::Error::msg(ACCESS_TOKEN_RULE_ERROR);
    if !token.chars().all(|c| ('!'..='~').contains(&c)) {
        return Err(unsendable());
    }
    // Backstop: the check above already rules out every value this rejects.
    let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| unsendable())?;
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
    /// The hint as plain lines, the start command (indented) included. This
    /// is the text [`ConnectHint::eprint`] prints, so the tests that check
    /// it check what `run()` shows.
    fn lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.details.len() + 2);
        lines.push(self.headline.clone());
        lines.extend(self.details.iter().cloned());
        if self.start_command {
            lines.push(format!("  {START_SERVER_COMMAND}"));
        }
        lines
    }

    /// Print [`ConnectHint::lines`] to stderr, the headline red and bold and
    /// the start command, when there is one, cyan.
    fn eprint(&self) {
        let lines = self.lines();
        // With a start command there are at least two lines, so the
        // command is never the headline.
        let command_line = self.start_command.then(|| lines.len() - 1);
        for (i, line) in lines.iter().enumerate() {
            if i == 0 {
                eprintln!("{}", line.red().bold());
            } else if Some(i) == command_line {
                eprintln!("{}", line.bright_cyan());
            } else {
                eprintln!("{line}");
            }
        }
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

/// How one prompt ended.
#[derive(Debug)]
enum Outcome {
    /// The answer, complete
    Done(Message),
    /// The request or the stream failed
    Failed(anyhow::Error),
    /// The user cancelled the answer (Ctrl+C)
    Cancelled,
}

/// Runs `answer` until it ends or `cancel` completes, whichever comes first. A cancelled
/// answer is dropped and its connection closed; the server stops the chat on the
/// disconnect.
async fn stream_or_cancel(
    answer: impl std::future::Future<Output = anyhow::Result<Message>>,
    cancel: impl std::future::Future<Output = ()>,
) -> Outcome {
    let outcome = tokio::select! {
        result = answer => match result {
            Ok(message) => Outcome::Done(message),
            Err(e) => Outcome::Failed(e),
        },
        () = cancel => Outcome::Cancelled,
    };
    if matches!(outcome, Outcome::Cancelled) {
        // The connection belongs to a task of the HTTP client, which closes it once polled
        // after the dropped answer. Let it run now: the caller may block next (the prompt
        // of the interactive mode reads the terminal on this thread).
        for _ in 0..CLOSE_YIELDS {
            tokio::task::yield_now().await;
        }
    }
    outcome
}

/// How often [`stream_or_cancel`] yields to the runtime after a cancellation.
const CLOSE_YIELDS: usize = 16;

/// Completes at the next Ctrl+C; never when the signal handler cannot be installed.
async fn ctrl_c() {
    if tokio::signal::ctrl_c().await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Run the chat: a script (`--script`) or the interactive prompt. A scripted run stops at
/// the first failed prompt and returns an error (with `--keep-going` it sends the rest and
/// still returns one); Ctrl+C cancels the current answer, and stops a scripted run.
pub async fn run(args: &ChatArgs) -> anyhow::Result<()> {
    let token = args
        .access_token
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty());
    let client = build_http_client(token)?;
    run_with_client(args, client, token.is_some()).await
}

/// [`run`] with the HTTP client given.
async fn run_with_client(
    args: &ChatArgs,
    client: reqwest::Client,
    token_sent: bool,
) -> anyhow::Result<()> {
    let base_url = args.url.trim_end_matches('/');

    // Create session
    let session_id = match create_session(&client, base_url, token_sent).await {
        Ok(id) => id,
        Err(err) => {
            connect_failure_hint(&err).eprint();
            return Err(anyhow::Error::new(err));
        }
    };

    // Initialize recorder
    let mut recorder = if let Some(ref path) = args.record {
        let mut rec = Recorder::new(path)?;
        rec.write_banner(args, base_url)?;
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
        session_label(&session_id),
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
    let mut charts = ChartNamer::default();

    // One prompt: send it, stream the answer (Ctrl+C cancels it) and extend the history
    // when the answer is complete.
    macro_rules! ask {
        ($input:expr) => {{
            let input: &str = $input;
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
            let outcome = stream_or_cancel(
                stream_chat(
                    &client,
                    base_url,
                    &request,
                    &args.output_dir,
                    &mut recorder,
                    &mut charts,
                    token_sent,
                ),
                ctrl_c(),
            )
            .await;
            match &outcome {
                Outcome::Done(msg) => {
                    history.push(Message {
                        role: "user".to_string(),
                        content: input.to_string(),
                        tool_calls: None,
                        tool_results: None,
                    });
                    history.push(msg.clone());
                }
                Outcome::Failed(e) => eprintln!("{} {}", "Request failed:".red(), e),
                Outcome::Cancelled => {
                    println!();
                    println!("{}", "Cancelled.".dimmed());
                    if let Some(ref mut rec) = recorder {
                        let _ = rec.write_cancelled();
                    }
                }
            }
            outcome
        }};
    }

    // Choose input source: script file or interactive readline
    if let Some(prompts) = script_prompts {
        let n = prompts.len();
        let mut failed = 0;
        for (k, input) in prompts.iter().enumerate() {
            // Echo the prompt (looks like interactive)
            println!("{} {}", "p2a>".bright_green().bold(), input);
            println!();
            match ask!(input) {
                Outcome::Done(_) => {}
                Outcome::Failed(e) => {
                    failed += 1;
                    if !args.keep_going {
                        anyhow::bail!(
                            "prompt {} of {n} failed: {e}; the remaining prompts were not sent (use --keep-going to send them)",
                            k + 1
                        );
                    }
                }
                Outcome::Cancelled => anyhow::bail!("interrupted at prompt {} of {n}", k + 1),
            }
        }
        if failed > 0 {
            anyhow::bail!("{failed} of {n} prompts failed");
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

            // a failed or cancelled answer returns to the prompt
            let _ = ask!(input);
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
    output_dir: &std::path::Path,
    recorder: &mut Option<Recorder>,
    charts: &mut ChartNamer,
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
    let mut last_error: Option<String> = None;

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
                match event {
                    StreamEvent::Done { ref message } => final_message = Some(message.clone()),
                    StreamEvent::Error { ref error } => last_error = Some(error.clone()),
                    _ => {}
                }
                render_event(&event, output_dir, recorder, charts);
            }
        }
    }

    // Process any trailing bytes with no final newline.
    if !buffer.is_empty() {
        let line = String::from_utf8_lossy(&buffer).into_owned();
        if let Some(event) = parse_sse_line(&line) {
            match event {
                StreamEvent::Done { ref message } => final_message = Some(message.clone()),
                StreamEvent::Error { ref error } => last_error = Some(error.clone()),
                _ => {}
            }
            render_event(&event, output_dir, recorder, charts);
        }
    }

    // the server's own error, when it sent one, says why there is no answer
    final_message.ok_or_else(|| match last_error {
        Some(e) => anyhow::Error::msg(e),
        None => anyhow::anyhow!("Stream ended without a Done event"),
    })
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
    println!("  - Ctrl+C cancels the current answer; Ctrl+D or /quit exits");
    println!();
    println!("{}", "Recording:".bold());
    println!("  --record file.txt   Save transcript to file");
    println!("  --script file.txt   Read prompts from file");
    println!("  --keep-going        With --script, send the remaining prompts after a failed one");
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
        // The command line comes from the flag (eprint() highlights it), so
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
        let help = cmd
            .get_arguments()
            .find(|a| a.get_id() == "access_token")
            .and_then(|a| a.get_help())
            .map(ToString::to_string)
            .unwrap_or_default();
        assert!(
            help.contains("visible ASCII, no spaces"),
            "the help states the token rule: {help:?}"
        );
        let args = ChatArgs::try_parse_from(["p2a-chat", "--access-token", "t0k"]).unwrap();
        assert_eq!(args.access_token.as_deref(), Some("t0k"));
    }

    #[test]
    fn token_with_control_characters_is_refused_before_sending() {
        let err = build_http_client(Some("bad\r\ntoken"))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err, ACCESS_TOKEN_RULE_ERROR,
            "the message names the source and never the value"
        );
    }

    /// Non-ASCII characters, inner spaces and tabs all pass
    /// `HeaderValue::from_str`. Of the three, the server's `to_str` rejects
    /// only non-ASCII; inner spaces and tabs it would compare literally. The
    /// client refuses all three to accept exactly the tokens the Dioxus app
    /// accepts. DEL, also in the list, already fails `HeaderValue::from_str`.
    #[test]
    fn token_outside_visible_ascii_is_refused_before_sending() {
        for token in ["tök", "a b", "a\tb", "x\u{7f}y"] {
            let err = build_http_client(Some(token)).unwrap_err().to_string();
            assert_eq!(err, ACCESS_TOKEN_RULE_ERROR, "{token:?}");
            assert!(!err.contains(token), "{token:?}");
        }
        // Surrounding whitespace is trimmed, not refused.
        assert!(build_http_client(Some("  ok-token\t")).is_ok());
    }

    /// CLI-1: the old `&s[..27]` cut inside 'é' (bytes 26..28) and crashed the client mid-stream.
    /// Expected strings: the p2a_text::ellipsize contract (T3-WP03 plan, Appendix A).
    #[test]
    fn format_args_brief_truncates_on_char_boundaries() {
        assert_eq!(
            format_args_brief(
                &serde_json::json!({"path": "/home/analyst/data/revenu_été_région_2024.csv"})
            ),
            "(path=\"/home/analyst/data/revenu_é...\")"
        );
        // non-string branch: the compact JSON has 'ä' at bytes 26..28
        assert_eq!(
            format_args_brief(
                &serde_json::json!({"x_cols": ["Größe_kg", "Gewicht_kg_äquivalent"]})
            ),
            "(x_cols=[\"Größe_kg\",\"Gewicht_kg_äqu...)"
        );
        // ASCII output is unchanged (the reviewer's control run printed exactly this)
        assert_eq!(
            format_args_brief(
                &serde_json::json!({"path": "/home/analyst/data/revenue_summer_2024.csv"})
            ),
            "(path=\"/home/analyst/data/revenue_...\")"
        );
        assert_eq!(
            format_args_brief(&serde_json::json!({"x": (1..=15).collect::<Vec<i32>>()})),
            "(x=[1,2,3,4,5,6,7,8,9,10,11,12...)"
        );
        assert_eq!(
            format_args_brief(&serde_json::json!({"dataset": "Größe"})),
            "(dataset=\"Größe\")"
        );
        assert_eq!(format_args_brief(&serde_json::json!("not an object")), "");
    }

    #[test]
    fn session_label_keeps_whole_characters() {
        assert_eq!(session_label("aéééé"), "aéééé"); // old [..8] cut inside 'é' (bytes 7..9)
        assert_eq!(
            session_label("0f8c2d4e-1b2a-4c3d-9e8f-001122334455"),
            "0f8c2d4e"
        );
        assert_eq!(session_label("abc"), "abc");
    }

    #[test]
    fn parse_sse_line_reads_data_events() {
        match parse_sse_line(
            r#"data: {"type":"tool_start","tool":"regression_ols","arguments":{"dataset":"g"}}"#,
        ) {
            Some(StreamEvent::ToolStart { tool, arguments }) => {
                assert_eq!(tool, "regression_ols");
                assert_eq!(arguments["dataset"], "g");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            parse_sse_line("  data: {\"type\":\"content\",\"text\":\"Größe ²\"}  "),
            Some(StreamEvent::Content { text }) if text == "Größe ²"
        ));
        assert!(matches!(
            parse_sse_line(
                r#"data: {"type":"done","message":{"role":"assistant","content":"ok"}}"#
            ),
            Some(StreamEvent::Done { .. })
        ));
        assert!(matches!(
            parse_sse_line(r#"data: {"type":"error","error":"boom"}"#),
            Some(StreamEvent::Error { error }) if error == "boom"
        ));
        assert!(matches!(
            parse_sse_line(r#"data: {"type":"tool_end","tool":"t","elapsed_ms":12}"#),
            Some(StreamEvent::ToolEnd { elapsed_ms: 12, .. })
        ));
    }

    #[test]
    fn parse_sse_line_skips_comments_other_fields_and_bad_json() {
        for line in [
            "",
            "   ",
            ":",
            ": keep-alive",
            "event: message",
            "id: 3",
            "retry: 100",
            "data: not json",
            r#"data: {"type":"unknown_kind"}"#,
            r#"data: {"type":"content"}"#,
        ] {
            assert!(parse_sse_line(line).is_none(), "{line:?}");
        }
    }

    /// A script in the temporary directory, removed when dropped (also when an assertion
    /// fails first).
    struct ScriptFile(PathBuf);

    impl Drop for ScriptFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn script_file(tag: &str, body: &str) -> ScriptFile {
        let p = std::env::temp_dir().join(format!(
            "p2a_chat_read_script_{}_{tag}.txt",
            std::process::id()
        ));
        std::fs::write(&p, body).unwrap();
        ScriptFile(p)
    }

    #[test]
    fn read_script_splits_prompts_on_blank_lines() {
        let p = script_file(
            "blank",
            "first prompt\n\nsecond prompt\n  \t \n  padded  \n\n\nlast",
        );
        assert_eq!(
            read_script(&p.0).unwrap(),
            ["first prompt", "second prompt", "padded", "last"]
        );
        let p = script_file("crlf", "a\r\n\r\nGröße of x?\r\n");
        assert_eq!(read_script(&p.0).unwrap(), ["a", "Größe of x?"]);
        let p = script_file("empty", "");
        assert!(read_script(&p.0).unwrap().is_empty());
        drop(p);
        assert!(read_script(&std::env::temp_dir().join("p2a_chat_no_such_script.txt")).is_err());
    }

    /// A scratch directory under the temporary directory, removed when dropped.
    struct ScratchDir(PathBuf);

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch_dir(tag: &str) -> ScratchDir {
        let p = std::env::temp_dir().join(format!("p2a_chat_charts_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        ScratchDir(p)
    }

    fn image_event(tool: Option<&str>, mime: &str, bytes: &[u8]) -> StreamEvent {
        StreamEvent::ToolResult {
            tool: tool.map(str::to_string),
            images: Some(vec![ImageData {
                data: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes),
                mime_type: mime.to_string(),
            }]),
        }
    }

    fn files(dir: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn charts_get_a_running_number_and_the_tool_name() {
        let d = scratch_dir("numbers");
        let mut charts = ChartNamer::default();
        let mut rec = None;
        render_event(
            &image_event(None, "image/png", b"first"),
            &d.0,
            &mut rec,
            &mut charts,
        );
        render_event(
            &image_event(Some("viz_scatter"), "image/png", b"second"),
            &d.0,
            &mut rec,
            &mut charts,
        );
        assert_eq!(files(&d.0), ["chart_001.png", "chart_002_viz_scatter.png"]);
        assert_eq!(std::fs::read(d.0.join("chart_001.png")).unwrap(), b"first");
        assert_eq!(
            std::fs::read(d.0.join("chart_002_viz_scatter.png")).unwrap(),
            b"second"
        );
    }

    #[test]
    fn charts_never_overwrite_a_file() {
        let d = scratch_dir("exists");
        std::fs::write(d.0.join("chart_001.png"), b"mine").unwrap();
        let mut charts = ChartNamer::default();
        render_event(
            &image_event(None, "image/png", b"new"),
            &d.0,
            &mut None,
            &mut charts,
        );
        assert_eq!(std::fs::read(d.0.join("chart_001.png")).unwrap(), b"mine");
        assert_eq!(std::fs::read(d.0.join("chart_002.png")).unwrap(), b"new");
    }

    #[test]
    fn chart_extensions_and_tool_names_are_safe() {
        let d = scratch_dir("names");
        let mut charts = ChartNamer::default();
        render_event(
            &image_event(None, "image/jpeg", b"j"),
            &d.0,
            &mut None,
            &mut charts,
        );
        render_event(
            &image_event(Some("../x y"), "image/png", b"p"),
            &d.0,
            &mut None,
            &mut charts,
        );
        render_event(
            &image_event(Some(""), "image/x-unknown", b"u"),
            &d.0,
            &mut None,
            &mut charts,
        );
        assert_eq!(
            files(&d.0),
            ["chart_001.jpg", "chart_002____x_y.png", "chart_003.bin"]
        );
        assert_eq!(chart_extension("image/jpg"), "jpg");
        assert_eq!(chart_extension("image/gif"), "gif");
        assert_eq!(chart_extension("image/webp"), "webp");
        assert_eq!(chart_extension("image/svg+xml"), "svg");
        assert_eq!(chart_tool_part(&"a".repeat(60)).len(), 40);
    }

    /// Reads one HTTP request (headers and a `Content-Length` body) from `stream`.
    fn read_request(stream: &mut std::net::TcpStream) -> String {
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
        String::from_utf8_lossy(&raw).into_owned()
    }

    /// Serves `responses` in order, one accepted connection per request, each answered
    /// with `Connection: close`; returns the URL and the number of requests served so far.
    fn sequential_server(
        responses: Vec<String>,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = count.clone();
        std::thread::spawn(move || {
            for response in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                read_request(&mut stream);
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}"), count)
    }

    fn json_response(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn sse_response(events: &[&str]) -> String {
        let mut r = String::from(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        );
        for e in events {
            r.push_str(&format!("data: {e}\n\n"));
        }
        r
    }

    const FAILED_ANSWER: &str = r#"{"type":"error","error":"provider down"}"#;
    const GOOD_ANSWER: [&str; 2] = [
        r#"{"type":"content","text":"ok"}"#,
        r#"{"type":"done","message":{"role":"assistant","content":"ok"}}"#,
    ];

    fn script_args(
        url: &str,
        script: &std::path::Path,
        out: &std::path::Path,
        extra: &[&str],
    ) -> ChatArgs {
        let mut argv = vec![
            "p2a-chat",
            "--url",
            url,
            "--script",
            script.to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--provider",
            "ollama",
            "--model",
            "m",
        ];
        argv.extend_from_slice(extra);
        ChatArgs::parse_from(argv)
    }

    async fn run_local(args: &ChatArgs) -> anyhow::Result<()> {
        let client = client_builder(None).unwrap().no_proxy().build().unwrap();
        run_with_client(args, client, false).await
    }

    #[tokio::test]
    async fn scripted_run_stops_at_a_failed_prompt() {
        let d = scratch_dir("stop");
        let script = script_file("stop", "first\n\nsecond\n");
        let (url, count) = sequential_server(vec![
            json_response(OK_SESSION),
            sse_response(&[FAILED_ANSWER]),
            sse_response(&GOOD_ANSWER),
        ]);
        let err = run_local(&script_args(&url, &script.0, &d.0, &[]))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("prompt 1 of 2 failed: provider down"),
            "{err}"
        );
        assert!(err.contains("--keep-going"), "{err}");
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn scripted_run_with_keep_going_sends_the_rest_and_still_fails() {
        let d = scratch_dir("keep");
        let script = script_file("keep", "first\n\nsecond\n");
        let (url, count) = sequential_server(vec![
            json_response(OK_SESSION),
            sse_response(&[FAILED_ANSWER]),
            sse_response(&[FAILED_ANSWER]),
        ]);
        let err = run_local(&script_args(&url, &script.0, &d.0, &["--keep-going"]))
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(err, "2 of 2 prompts failed");
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn scripted_run_with_good_answers_succeeds() {
        let d = scratch_dir("good");
        let script = script_file("good", "first\n\nsecond\n");
        let (url, count) = sequential_server(vec![
            json_response(OK_SESSION),
            sse_response(&GOOD_ANSWER),
            sse_response(&GOOD_ANSWER),
        ]);
        run_local(&script_args(&url, &script.0, &d.0, &[]))
            .await
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn stream_or_cancel_returns_cancelled() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (fire, cancel) = futures::channel::oneshot::channel::<()>();
        let (closed_tx, closed_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            read_request(&mut stream);
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(b"data: {\"type\":\"content\",\"text\":\"partial\"}\n\n");
            let _ = stream.flush();
            let _ = fire.send(());
            // hang guard, not a timing assertion: wait up to 10 s for the client to close
            stream
                .set_read_timeout(Some(std::time::Duration::from_millis(100)))
                .unwrap();
            let mut buf = [0u8; 64];
            for _ in 0..100 {
                match stream.read(&mut buf) {
                    Ok(0) => {
                        let _ = closed_tx.send(true);
                        return;
                    }
                    Ok(_) => {}
                    Err(e)
                        if matches!(
                            e.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) => {}
                    Err(_) => {
                        let _ = closed_tx.send(true);
                        return;
                    }
                }
            }
            let _ = closed_tx.send(false);
        });
        let client = client_builder(None).unwrap().no_proxy().build().unwrap();
        let request = LlmChatRequest {
            session_id: "s".into(),
            message: "hi".into(),
            provider: None,
            history: None,
            interpret: true,
            conversation_id: None,
            retrieve_history: false,
        };
        let d = scratch_dir("cancel");
        let mut recorder = None;
        let mut charts = ChartNamer::default();
        let outcome = stream_or_cancel(
            stream_chat(
                &client,
                &url,
                &request,
                &d.0,
                &mut recorder,
                &mut charts,
                false,
            ),
            async {
                let _ = cancel.await;
            },
        )
        .await;
        assert!(matches!(outcome, Outcome::Cancelled), "{outcome:?}");
        assert!(
            closed_rx
                .recv_timeout(std::time::Duration::from_secs(15))
                .unwrap(),
            "the client closed the connection"
        );
    }

    #[test]
    fn banner_records_the_command() {
        let d = scratch_dir("banner");
        let record = d.0.join("t.txt");
        let args = ChatArgs::parse_from([
            "p2a-chat",
            "--url",
            "http://127.0.0.1:9/x",
            "--provider",
            "openai",
            "--model",
            "gpt x",
            "--base-url",
            "http://h/v1",
            "--script",
            "p.txt",
            "--api-key",
            "sk-test-NOTRECORDED-7f3",
            "--access-token",
            "tok-NOTRECORDED-9c1",
            "--record",
            record.to_str().unwrap(),
        ]);
        let mut rec = Recorder::new(&record).unwrap();
        rec.write_banner(&args, "http://127.0.0.1:9/x").unwrap();
        drop(rec);
        let text = std::fs::read_to_string(&record).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap(),
            format!(
                "$ p2a chat --url http://127.0.0.1:9/x --provider openai --model 'gpt x' --base-url http://h/v1 --script p.txt --record {}",
                quote(record.to_str().unwrap())
            )
        );
        assert_eq!(
            lines.next().unwrap(),
            "# --api-key / --access-token given, not recorded"
        );
        assert!(!text.contains("NOTRECORDED"), "{text}");
        assert_eq!(quote("it's"), r"'it'\''s'");
        assert_eq!(quote(""), "''");
    }

    #[test]
    fn read_script_keeps_hash_prompts_and_line_breaks() {
        let p = script_file(
            "hash",
            "# note\n\\#3: rerun with clustering\n\nline one\nline two\n",
        );
        assert_eq!(
            read_script(&p.0).unwrap(),
            ["#3: rerun with clustering", "line one\nline two"]
        );
    }
}
