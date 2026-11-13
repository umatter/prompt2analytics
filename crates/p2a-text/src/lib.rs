//! Char-boundary-safe text truncation for prompt2analytics.
//!
//! Slicing a `&str` at a byte offset (`&s[..n]`) panics when byte `n` falls inside a multi-byte
//! UTF-8 character: a column called `Größe_kg`, a CJK dataset name or a path with an accent is
//! enough. In the server such a panic kills the request task, in the CLI and the chat client it
//! ends the process. Every truncation of user, dataset or LLM text in the workspace therefore goes
//! through this crate (rule R8 of the October 2026 review) instead of a byte slice.
//!
//! Two families of functions:
//!
//! - **Byte budgets** ([`floor_char_boundary`], [`truncate_bytes`], [`ellipsize_bytes`]) for
//!   storage, logs and protocol limits, where the size in bytes is what is bounded.
//! - **Character budgets** ([`truncate_chars`], [`ellipsize`]) for display columns, where a name
//!   is shortened to a number of characters.
//!
//! "Character" means a Unicode scalar value (`char`), not a grapheme cluster: a combining mark can
//! be separated from its base character, but a UTF-8 sequence is never split, so every result is
//! valid UTF-8 and no function here panics. The crate has no dependencies, uses only `core`/`std`
//! string APIs and builds for `wasm32-unknown-unknown`.
//!
//! [`floor_char_boundary`] predates `str::floor_char_boundary`, which was not stable at the
//! workspace's earlier MSRV (1.89); it stays part of this crate's API.
#![forbid(unsafe_code)]

use std::borrow::Cow;

/// The largest char boundary of `s` that is at most `max_bytes`; `s.len()` when `max_bytes >= s.len()`.
///
/// ```
/// assert_eq!(p2a_text::floor_char_boundary("Größe_kg", 5), 4); // 'ß' is bytes 4..6
/// ```
pub fn floor_char_boundary(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() {
        return s.len();
    }
    // 0 is always a boundary, and a UTF-8 sequence is at most 4 bytes: at most 3 steps.
    let mut i = max_bytes;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The longest prefix of `s` that is at most `max_bytes` bytes long and ends on a char boundary.
///
/// ```
/// assert_eq!(p2a_text::truncate_bytes("収入合計額", 7), "収入");
/// ```
pub fn truncate_bytes(s: &str, max_bytes: usize) -> &str {
    &s[..floor_char_boundary(s, max_bytes)]
}

/// The first `max_chars` characters (Unicode scalar values) of `s`; `s` when it has no more.
///
/// ```
/// assert_eq!(p2a_text::truncate_chars("Größe_kg", 3), "Grö");
/// ```
pub fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Display shortening by characters: if `s` has more than `max_chars` characters, its first
/// `max_chars - 3` characters followed by `"..."` (`max_chars` characters in all); `s` unchanged
/// otherwise. With `max_chars < 4` there is no room for the ellipsis and this is
/// [`truncate_chars`]. For ASCII text it prints exactly what `&s[..max_chars - 3] + "..."` did.
///
/// ```
/// assert_eq!(p2a_text::ellipsize("Haushaltsgrößenklasse", 15), "Haushaltsgrö...");
/// assert_eq!(p2a_text::ellipsize("Größe_kg", 8), "Größe_kg");
/// ```
pub fn ellipsize(s: &str, max_chars: usize) -> Cow<'_, str> {
    if s.char_indices().nth(max_chars).is_none() {
        return Cow::Borrowed(s);
    }
    if max_chars < 4 {
        return Cow::Borrowed(truncate_chars(s, max_chars));
    }
    Cow::Owned(format!("{}...", truncate_chars(s, max_chars - 3)))
}

/// Byte-budget shortening with a marker counted inside the budget: `s` unchanged if
/// `s.len() <= max_bytes`; `""` if the marker alone does not fit; otherwise
/// `truncate_bytes(s, max_bytes - marker.len())` followed by `marker`. The result is never longer
/// than `max_bytes` bytes.
///
/// ```
/// let s = "ä".repeat(20);
/// assert_eq!(p2a_text::ellipsize_bytes(&s, 21, "\n[truncated]"), "ääää\n[truncated]");
/// ```
pub fn ellipsize_bytes<'a>(s: &'a str, max_bytes: usize, marker: &str) -> Cow<'a, str> {
    if s.len() <= max_bytes {
        return Cow::Borrowed(s);
    }
    let Some(room) = max_bytes.checked_sub(marker.len()) else {
        return Cow::Borrowed("");
    };
    let mut out = String::with_capacity(room + marker.len());
    out.push_str(truncate_bytes(s, room));
    out.push_str(marker);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    const SAMPLES: [&str; 9] = [
        "",
        "abc",
        "Größe_kg",
        "収入合計額",
        "R²",
        "😀x",
        "Haushaltsgrößenklasse",
        "a\u{301}bc",
        "/home/analyst/data/revenu_été_région_2024.csv",
    ];

    #[test]
    fn floor_char_boundary_cases() {
        assert_eq!(floor_char_boundary("", 0), 0);
        assert_eq!(floor_char_boundary("", 5), 0);
        assert_eq!(floor_char_boundary("abc", 10), 3);
        assert_eq!(floor_char_boundary("abc", 3), 3); // exact boundary
        assert_eq!(floor_char_boundary("abc", 2), 2);
        assert_eq!(floor_char_boundary("aé", 2), 1); // 'é' is bytes 1..3
        assert_eq!(floor_char_boundary("aé", 3), 3);
        assert_eq!(floor_char_boundary("😀x", 1), 0); // 4-byte emoji, bytes 0..4
        assert_eq!(floor_char_boundary("😀x", 3), 0);
        assert_eq!(floor_char_boundary("😀x", 4), 4);
        assert_eq!(floor_char_boundary("😀x", 5), 5);
        assert_eq!(floor_char_boundary("Größe_kg", 5), 4); // 'ß' is bytes 4..6
        assert_eq!(floor_char_boundary("収入合計額", 7), 6); // 3-byte CJK
        assert_eq!(floor_char_boundary("R²", 2), 1); // '²' is bytes 1..3
        assert_eq!(floor_char_boundary("abc", usize::MAX), 3);
    }

    #[test]
    fn truncate_bytes_cases() {
        assert_eq!(truncate_bytes("Größe_kg", 5), "Grö");
        assert_eq!(truncate_bytes("Größe_kg", 6), "Größ");
        assert_eq!(truncate_bytes("Größe_kg", 0), ""); // max 0
        assert_eq!(truncate_bytes("Größe_kg", 100), "Größe_kg");
        assert_eq!(truncate_bytes("収入合計額", 7), "収入");
        assert_eq!(truncate_bytes("R²", 2), "R");
        assert_eq!(truncate_bytes("😀x", 3), "");
        assert_eq!(truncate_bytes("", 0), "");
    }

    #[test]
    fn truncate_chars_cases() {
        assert_eq!(truncate_chars("Größe_kg", 3), "Grö");
        assert_eq!(truncate_chars("収入合計額", 2), "収入");
        assert_eq!(truncate_chars("😀😀", 1), "😀");
        assert_eq!(truncate_chars("abc", 0), ""); // max 0
        assert_eq!(truncate_chars("", 3), "");
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abc", 10), "abc");
        // Unicode scalar values, not grapheme clusters: a combining mark may be split off,
        // a UTF-8 sequence never.
        assert_eq!(truncate_chars("a\u{301}bc", 1), "a");
    }

    #[test]
    fn ellipsize_cases() {
        assert!(matches!(
            ellipsize("Größe_kg", 8),
            Cow::Borrowed("Größe_kg")
        )); // 8 chars: unchanged
        assert_eq!(ellipsize("Größe_kg", 7), "Größ...");
        assert_eq!(ellipsize("収入合計額", 4), "収...");
        assert_eq!(ellipsize("Haushaltsgrößenklasse", 15), "Haushaltsgrö...");
        assert_eq!(ellipsize("R² = 0.95", 6), "R² ...");
        assert_eq!(ellipsize("abcdef", 3), "abc"); // max < 4: plain truncation
        assert_eq!(ellipsize("abcdef", 0), ""); // max 0
        assert_eq!(ellipsize("", 0), ""); // empty
        assert!(matches!(ellipsize("abc", 5), Cow::Borrowed("abc")));
        // ASCII behaves as the old byte rule `if len > m { &s[..m - 3] + "..." }` did:
        assert_eq!(
            ellipsize("/home/analyst/data/revenue_summer_2024.csv", 30),
            "/home/analyst/data/revenue_..."
        );
    }

    #[test]
    fn ellipsize_bytes_cases() {
        let mk = "\n[truncated]"; // 12 bytes
        assert_eq!(
            ellipsize_bytes(&"ä".repeat(20), 20, mk),
            "ääää\n[truncated]"
        ); // 8 + 12 = 20 bytes
        assert_eq!(
            ellipsize_bytes(&"ä".repeat(20), 21, mk),
            "ääää\n[truncated]"
        ); // room 9 floors to 8
        assert!(matches!(
            ellipsize_bytes("short", 100, mk),
            Cow::Borrowed("short")
        ));
        assert_eq!(ellipsize_bytes(&"ä".repeat(20), 5, mk), ""); // marker does not fit
        assert_eq!(ellipsize_bytes(&"ä".repeat(20), 12, mk), "\n[truncated]"); // marker exactly fits
        assert_eq!(ellipsize_bytes("😀😀😀", 7, "~"), "😀~"); // 5 bytes
        assert_eq!(ellipsize_bytes("", 0, "x"), "");
        assert_eq!(ellipsize_bytes("abc", 0, ""), "");
    }

    #[test]
    fn every_cut_point_is_safe_and_maximal() {
        for s in SAMPLES {
            let n_chars = s.chars().count();
            for m in 0..=s.len() + 2 {
                let b = truncate_bytes(s, m);
                assert!(s.starts_with(b) && b.len() <= m, "{s:?} {m}");
                assert!(b.len() + 3 >= m.min(s.len()), "not maximal: {s:?} {m}"); // at most 3 bytes short
                assert_eq!(floor_char_boundary(s, m), b.len());
                let c = truncate_chars(s, m);
                assert!(
                    s.starts_with(c) && c.chars().count() == m.min(n_chars),
                    "{s:?} {m}"
                );
                let e = ellipsize(s, m);
                if n_chars <= m {
                    assert_eq!(e, s)
                } else if m >= 4 {
                    assert!(
                        e.ends_with("...") && e.chars().count() == m,
                        "{s:?} {m} {e:?}"
                    );
                    assert!(
                        s.starts_with(e.strip_suffix("...").unwrap()),
                        "{s:?} {m} {e:?}"
                    );
                } else {
                    assert_eq!(e, truncate_chars(s, m))
                }
                let eb = ellipsize_bytes(s, m, "~");
                if s.len() <= m {
                    assert_eq!(eb, s)
                } else {
                    assert!(eb.len() <= m, "{s:?} {m} {eb:?}")
                }
            }
        }
    }
}
