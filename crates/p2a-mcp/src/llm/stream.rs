//! The line buffer the providers' stream parsers share: Anthropic's and OpenAI's server-sent
//! events and Ollama's NDJSON are all lines of UTF-8 text, delivered in network reads that may end
//! anywhere, inside a line or inside a character.
//!
//! [`LineBuffer`] keeps the raw bytes and decodes a line only once it is complete, so a character
//! split across two reads is never decoded in halves. Consumed bytes are dropped by an offset and
//! the buffer is compacted at most once per [`LineBuffer::push`], only when the consumed part is at
//! least as long as what is left, and the search for the next line break resumes where the last one
//! stopped: the work is linear in the stream, however the reads split it.

use super::LlmError;

/// Raw bytes of a provider stream, handed out as complete UTF-8 lines.
#[derive(Debug, Default)]
pub(crate) struct LineBuffer {
    buf: Vec<u8>,
    /// Start of the bytes not yet handed out.
    start: usize,
    /// Bytes from `start` on already searched for a line break (none found).
    scanned: usize,
}

impl LineBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append the bytes of one network read.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        // Drop the consumed bytes when they are at least half the buffer: each byte is moved a
        // bounded number of times, whatever the reads look like.
        if self.start > 0 && self.start >= self.buf.len() - self.start {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.extend_from_slice(bytes);
    }

    /// The next complete line, without its `\n` and one `\r` before it; `None` until a line break
    /// arrives. A line that is not UTF-8 is an error.
    pub(crate) fn next_line(&mut self) -> Option<Result<String, LlmError>> {
        let rest = &self.buf[self.start..];
        let Some(found) = rest[self.scanned..].iter().position(|&b| b == b'\n') else {
            self.scanned = rest.len();
            return None;
        };
        let pos = self.scanned + found;
        let line = decode(&rest[..pos]);
        self.start += pos + 1;
        self.scanned = 0;
        Some(line)
    }

    /// At the end of the stream: the last line when it has no `\n` (`None` when nothing is left).
    pub(crate) fn finish(&mut self) -> Option<Result<String, LlmError>> {
        let rest = &self.buf[self.start..];
        let line = (!rest.is_empty()).then(|| decode(rest));
        self.buf.clear();
        self.start = 0;
        self.scanned = 0;
        line
    }
}

/// A line's text without one trailing `\r`.
fn decode(line: &[u8]) -> Result<String, LlmError> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    String::from_utf8(line.to_vec()).map_err(|_| {
        LlmError::SerializationError("the provider sent text that is not UTF-8".to_string())
    })
}

/// The payload of a server-sent event's `data:` line (one optional space after the colon); `None`
/// for any other line (`event:`, `id:`, a `:` comment, a blank line).
pub(crate) fn sse_data(line: &str) -> Option<&str> {
    let data = line.strip_prefix("data:")?;
    Some(data.strip_prefix(' ').unwrap_or(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every line `buf` hands out after `pushes`, then the one `finish` gives.
    fn lines(pushes: &[&[u8]]) -> (Vec<String>, Option<String>) {
        let mut buf = LineBuffer::new();
        let mut out = Vec::new();
        for p in pushes {
            buf.push(p);
            while let Some(line) = buf.next_line() {
                out.push(line.unwrap());
            }
        }
        (out, buf.finish().map(Result::unwrap))
    }

    #[test]
    fn a_character_split_across_reads_is_decoded_whole() {
        // 'ö' is C3 B6 (2 bytes), '€' is E2 82 AC (3 bytes).
        let (got, rest) = lines(&[b"gr\xC3", b"\xB6\xC3\x9Fe \xE2", b"\x82", b"\xAC\n"]);
        assert_eq!((got, rest), (vec!["größe €".to_string()], None));
    }

    #[test]
    fn crlf_and_a_line_over_three_reads() {
        let (got, rest) = lines(&[b"data: a\r\nda", b"ta", b": b\r\n\r\n"]);
        assert_eq!(got, ["data: a", "data: b", ""]);
        assert_eq!(rest, None);
    }

    #[test]
    fn finish_gives_a_last_line_without_a_line_break() {
        assert_eq!(
            lines(&[b"x\n{\"done\":", b"true}"]).1.as_deref(),
            Some("{\"done\":true}")
        );
        assert_eq!(lines(&[b"x\n"]).1, None);
        assert_eq!(lines(&[b"x\ny\r"]).1.as_deref(), Some("y"));
    }

    #[test]
    fn invalid_utf8_is_an_error() {
        let mut buf = LineBuffer::new();
        buf.push(b"ok\n\xFF\xFE\nnext\n");
        assert_eq!(buf.next_line().unwrap().unwrap(), "ok");
        let e = buf.next_line().unwrap().unwrap_err();
        assert!(e.to_string().contains("not UTF-8"), "{e}");
        assert_eq!(buf.next_line().unwrap().unwrap(), "next");
        buf.push(b"\xC3");
        assert!(buf.finish().unwrap().is_err());
    }

    /// Many short lines in reads that cut them anywhere come out as they went in, and the buffer
    /// never holds much more than the unread part.
    #[test]
    fn many_lines_in_odd_reads() {
        let text: String = (0..2_000).map(|i| format!("line {i} ü\n")).collect();
        let bytes = text.as_bytes();
        let mut buf = LineBuffer::new();
        let mut got = Vec::new();
        for chunk in bytes.chunks(7) {
            buf.push(chunk);
            while let Some(line) = buf.next_line() {
                got.push(line.unwrap());
            }
            assert!(buf.buf.len() <= 2 * 16 + 7, "{}", buf.buf.len());
        }
        assert_eq!(got.len(), 2_000);
        assert_eq!(got[1_999], "line 1999 ü");
    }

    /// A long line over many reads is searched once: `scanned` follows the end of the buffer.
    #[test]
    fn a_long_line_is_searched_once() {
        let mut buf = LineBuffer::new();
        for _ in 0..100 {
            buf.push(&[b'x'; 10]);
            assert!(buf.next_line().is_none());
            assert_eq!(buf.scanned, buf.buf.len() - buf.start);
        }
        buf.push(b"\n");
        assert_eq!(buf.next_line().unwrap().unwrap().len(), 1_000);
    }

    #[test]
    fn sse_data_payloads() {
        assert_eq!(sse_data("data: x"), Some("x"));
        assert_eq!(sse_data("data:x"), Some("x"));
        assert_eq!(sse_data("data:  x"), Some(" x"));
        assert_eq!(sse_data(": comment"), None);
        assert_eq!(sse_data("event: e"), None);
        assert_eq!(sse_data(""), None);
    }
}
