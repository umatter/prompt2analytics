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
    #[arg(long, env = "P2A_API_KEY")]
    pub api_key: Option<String>,

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
// REPL
// ---------------------------------------------------------------------------

pub async fn run(args: &ChatArgs) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let base_url = args.url.trim_end_matches('/');

    // Create session
    let session_id = match create_session(&client, base_url).await {
        Ok(id) => id,
        Err(e) => {
            eprintln!("{}", "Could not connect to p2a-mcp server.".red().bold());
            eprintln!("  {}", e);
            eprintln!();
            eprintln!("Start the server first:");
            eprintln!(
                "  {}",
                "cargo run -p p2a-mcp --features full -- --transport http --port 8080"
                    .bright_cyan()
            );
            return Err(e);
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

            match stream_chat(&client, base_url, &request, &args.output_dir, &mut recorder).await {
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

            match stream_chat(&client, base_url, &request, &args.output_dir, &mut recorder).await {
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

async fn create_session(client: &reqwest::Client, base_url: &str) -> anyhow::Result<String> {
    let resp = client
        .post(format!("{}/api/sessions", base_url))
        .json(&serde_json::json!({}))
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("Server returned {}", resp.status());
    }

    let body: ApiResponse<CreateSessionData> = resp.json().await?;
    let data = body
        .data
        .ok_or_else(|| anyhow::anyhow!("No data in session response"))?;
    Ok(data.session_id)
}

async fn stream_chat(
    client: &reqwest::Client,
    base_url: &str,
    request: &LlmChatRequest,
    output_dir: &PathBuf,
    recorder: &mut Option<Recorder>,
) -> anyhow::Result<Message> {
    let resp = client
        .post(format!("{}/api/llm/chat/stream", base_url))
        .json(request)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {}", status, body);
    }

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut final_message: Option<Message> = None;

    while let Some(chunk_result) = stream.next().await {
        let bytes = chunk_result?;
        let text = String::from_utf8_lossy(&bytes);
        buffer.push_str(&text);

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if let Some(event) = parse_sse_line(&line) {
                if let StreamEvent::Done { ref message } = event {
                    final_message = Some(message.clone());
                }
                render_event(&event, output_dir, recorder);
            }
        }
    }

    // Process remaining buffer
    if !buffer.is_empty() {
        if let Some(event) = parse_sse_line(&buffer) {
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
