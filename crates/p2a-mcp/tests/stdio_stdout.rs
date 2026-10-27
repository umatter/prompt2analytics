//! The stdio transport's stdout carries JSON-RPC messages and nothing else.
//!
//! These tests run the real `p2a-mcp` binary over stdin/stdout, as an MCP
//! client (Claude Desktop and the like) does. A tool whose code path writes to
//! the process's stdout, through a `println!` in p2a or in a dependency, puts
//! a line into the JSON-RPC stream that the client cannot parse. The
//! in-process stdio tests (`limits_e2e.rs`) serve a duplex stream and cannot
//! see such a line, because it goes to the real stdout.
//!
//! A test ends the session by closing stdin, as a client does, and reads
//! stdout until the process closes it, so the check covers every line the
//! process wrote: those of a background task or of the shutdown path, and
//! not only those before the last response.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How long to wait for the next stdout line, or for the process to exit,
/// before failing. This only keeps a hung server from hanging `cargo test`
/// (CI sets no job timeout); it asserts nothing about speed, and a line
/// normally arrives in well under a second.
const PATIENCE: Duration = Duration::from_secs(60);

/// A `p2a-mcp --transport stdio` child process and every line it has
/// written to stdout so far.
struct StdioServer {
    child: Child,
    stdin: Option<ChildStdin>,
    /// The child's stdout, one line at a time, from a reader thread: a
    /// line (newline included), `Ok(None)` at the end of stdout, or the
    /// read error. A blocking read here could hang the test for good.
    stdout: Receiver<io::Result<Option<String>>>,
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
        let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            loop {
                // Bytes, not `read_line`: a write that is not UTF-8 is a
                // stray line too, not a read error.
                let mut buf = Vec::new();
                let event = match stdout.read_until(b'\n', &mut buf) {
                    Ok(0) => Ok(None),
                    Ok(_) => Ok(Some(String::from_utf8_lossy(&buf).into_owned())),
                    Err(e) => Err(e),
                };
                let more = matches!(event, Ok(Some(_)));
                if tx.send(event).is_err() || !more {
                    return;
                }
            }
        });
        let mut server = StdioServer {
            child,
            stdin,
            stdout: rx,
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

    /// A panic naming `what` the test was waiting for, with the stdout
    /// lines read so far and the server's stderr.
    fn fail(&self, what: &str) -> ! {
        let stderr = std::fs::read_to_string(&self.stderr_log).unwrap_or_default();
        panic!(
            "p2a-mcp {what}; stdout so far: {:?}; stderr: {stderr}",
            self.lines
        );
    }

    /// The next stdout line, recorded in `lines`, or `None` once stdout has
    /// ended. Fails if no line arrives within `PATIENCE`; `waiting_for`
    /// says for what.
    fn next_line(&mut self, waiting_for: &str) -> Option<String> {
        match self.stdout.recv_timeout(PATIENCE) {
            Ok(Ok(Some(line))) => {
                self.lines.push(line.clone());
                Some(line)
            }
            // The reader thread hangs up only after the end or an error.
            Ok(Ok(None)) | Err(RecvTimeoutError::Disconnected) => None,
            Ok(Err(e)) => self.fail(&format!("stdout read failed ({e}) {waiting_for}")),
            Err(RecvTimeoutError::Timeout) => self.fail(&format!(
                "wrote nothing to stdout for {PATIENCE:?} {waiting_for}"
            )),
        }
    }

    /// Sends request `id` and returns its response, recording every stdout
    /// line read on the way, JSON or not.
    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let waiting_for = format!("while answering request {id}");
        loop {
            let Some(line) = self.next_line(&waiting_for) else {
                self.fail(&format!("closed stdout before answering request {id}"));
            };
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

    /// Ends the session as a client does, by closing stdin, records the
    /// rest of stdout up to its end, and returns the exit status once the
    /// process has exited.
    fn finish(&mut self) -> ExitStatus {
        drop(self.stdin.take());
        while self
            .next_line("after stdin closed, without closing stdout")
            .is_some()
        {}
        // Stdout has ended, so the process is exiting; poll rather than
        // `wait`, under the same guard against a hang.
        let deadline = Instant::now() + PATIENCE;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return status,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(None) => self.fail(&format!(
                    "closed stdout but had not exited {PATIENCE:?} later"
                )),
                Err(e) => self.fail(&format!("could not be waited for ({e})")),
            }
        }
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
        // After `finish` the child has exited and been reaped. A test that
        // failed before then leaves it running: kill it, so that neither
        // the process nor the reader thread outlives the test.
        drop(self.stdin.take());
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
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
/// answers with a JSON-RPC response and the process leaves nothing else on
/// stdout up to its exit: polars' `pivot` entry point prints a notice to
/// stdout on every call.
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
    let status = server.finish();
    assert!(
        server.stray_lines().is_empty(),
        "stdout lines that are not JSON-RPC: {:?}",
        server.stray_lines()
    );
    assert!(
        status.success(),
        "p2a-mcp exited with {status} after stdin closed"
    );
}
