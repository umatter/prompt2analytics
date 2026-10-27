//! The stdio transport's stdout carries JSON-RPC messages and nothing else.
//!
//! These tests run the real `p2a-mcp` binary over stdin/stdout, as an MCP
//! client (Claude Desktop and the like) does. A tool whose code path writes to
//! the process's stdout, through a `println!` in p2a or in a dependency, puts
//! a line into the JSON-RPC stream that the client cannot parse. The
//! in-process stdio tests (`limits_e2e.rs`) serve a duplex stream and cannot
//! see such a line, because it goes to the real stdout.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

/// A `p2a-mcp --transport stdio` child process and every line it has
/// written to stdout so far.
struct StdioServer {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_log: PathBuf,
    lines: Vec<String>,
}

impl StdioServer {
    /// Starts the binary with an empty environment apart from `HOME` and
    /// `P2A_DATA_ROOT`, both `root`, so that nothing it writes leaves `root`,
    /// and runs the MCP initialisation handshake.
    fn start(root: &Path) -> Self {
        let stderr_log = root.join("p2a-mcp.stderr");
        let mut child = Command::new(env!("CARGO_BIN_EXE_p2a-mcp"))
            .args(["--transport", "stdio"])
            .env_clear()
            .env("HOME", root)
            .env("P2A_DATA_ROOT", root)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(File::create(&stderr_log).expect("stderr log"))
            .spawn()
            .expect("start p2a-mcp");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut server = StdioServer {
            child,
            stdin,
            stdout,
            stderr_log,
            lines: Vec::new(),
        };
        server.request(
            1,
            "initialize",
            json!({"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "stdio-stdout-test", "version": "0"}}),
        );
        server.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        server
    }

    fn send(&mut self, msg: Value) {
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{msg}").expect("write to p2a-mcp");
        stdin.flush().expect("flush to p2a-mcp");
    }

    /// Sends request `id` and returns its response, recording every stdout
    /// line read on the way, JSON or not.
    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let mut line = String::new();
            let n = self
                .stdout
                .read_line(&mut line)
                .expect("read p2a-mcp stdout");
            if n == 0 {
                let stderr = std::fs::read_to_string(&self.stderr_log).unwrap_or_default();
                panic!(
                    "p2a-mcp closed stdout before answering request {id}; \
                     stdout so far: {:?}; stderr: {stderr}",
                    self.lines
                );
            }
            self.lines.push(line.clone());
            if let Ok(msg) = serde_json::from_str::<Value>(&line)
                && msg["id"] == id
            {
                return msg;
            }
        }
    }

    /// The `tools/call` response for `tool` with `args`.
    fn call(&mut self, id: u64, tool: &str, args: Value) -> Value {
        self.request(id, "tools/call", json!({"name": tool, "arguments": args}))
    }

    /// The stdout lines so far that are not a JSON-RPC 2.0 message.
    fn stray_lines(&self) -> Vec<&String> {
        self.lines
            .iter()
            .filter(|line| {
                serde_json::from_str::<Value>(line)
                    .map(|msg| msg["jsonrpc"] != "2.0")
                    .unwrap_or(true)
            })
            .collect()
    }
}

impl Drop for StdioServer {
    fn drop(&mut self) {
        // Closing stdin ends the stdio session; kill in case it does not.
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The text of a successful tool result, or a panic naming the response.
fn tool_text(response: &Value) -> &str {
    assert_eq!(response["result"]["isError"], false, "{response}");
    response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no text content: {response}"))
}

/// `munge_pivot` (two index columns, a null in `on`, a missing combination)
/// answers with a JSON-RPC response and leaves nothing else on stdout:
/// polars' `pivot` entry point prints a notice to stdout on every call.
#[test]
fn munge_pivot_writes_only_json_rpc_to_stdout() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut server = StdioServer::start(root.path());
    let csv = "id,grp,year,value\n\
               1,a,2020,100.3\n\
               1,a,2021,110.9\n\
               1,b,2020,98.1\n\
               2,a,2020,201.7\n\
               2,a,2021,219.4\n\
               2,b,,205.2\n";
    let created = server.call(
        2,
        "create_dataset",
        json!({"name": "long", "csv_content": csv}),
    );
    tool_text(&created);
    let pivoted = server.call(
        3,
        "munge_pivot",
        json!({"dataset": "long", "index": ["id", "grp"], "on": "year",
               "values": "value", "result_name": "wide"}),
    );
    // (id, grp) has 4 combinations; `year` has 2020, 2021 and a null.
    assert!(
        tool_text(&pivoted).contains("saved as 'wide' (4 rows, 5 columns)"),
        "{pivoted}"
    );
    assert!(
        server.stray_lines().is_empty(),
        "stdout lines that are not JSON-RPC: {:?}",
        server.stray_lines()
    );
}
