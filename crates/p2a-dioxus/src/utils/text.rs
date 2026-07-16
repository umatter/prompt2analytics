//! Text utilities.

/// Return the largest prefix of `s` that is at most `max_bytes` long and ends
/// on a UTF-8 character boundary.
///
/// Slicing a `&str` with a raw byte range (`&s[..max_bytes]`) panics when the
/// cut falls inside a multi-byte character — very common for analytics output
/// (σ, β, χ², ±, accented names, non-ASCII data values). This truncates safely
/// instead, never splitting a character.
pub fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk back to the nearest char boundary at or below max_bytes.
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::truncate_on_char_boundary;

    #[test]
    fn does_not_split_multibyte_char() {
        // "áé" is 4 bytes (2 each). Truncating to 3 bytes must not split 'é'.
        let s = "áé";
        assert_eq!(truncate_on_char_boundary(s, 3), "á");
        assert_eq!(truncate_on_char_boundary(s, 2), "á");
        assert_eq!(truncate_on_char_boundary(s, 1), "");
        assert_eq!(truncate_on_char_boundary(s, 4), "áé");
        assert_eq!(truncate_on_char_boundary(s, 99), "áé");
    }

    #[test]
    fn ascii_unchanged() {
        assert_eq!(truncate_on_char_boundary("hello", 3), "hel");
        assert_eq!(truncate_on_char_boundary("hi", 10), "hi");
    }
}
