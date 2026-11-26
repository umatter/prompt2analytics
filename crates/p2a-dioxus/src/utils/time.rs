//! Times shown in the app: in the viewer's local time zone (the web build reads it from the
//! browser through chrono's `wasmbind`).

use chrono::{DateTime, Local, Utc};

/// `HH:MM` of `t` in local time.
pub fn local_clock(t: &DateTime<Utc>) -> String {
    t.with_timezone(&Local).format("%H:%M").to_string()
}

/// An RFC 3339 timestamp as `YYYY-MM-DD HH:MM` in local time, to the minute; a value that is
/// not RFC 3339 is shown as it is.
pub fn local_stamp(rfc3339: &str) -> String {
    match DateTime::parse_from_rfc3339(rfc3339) {
        Ok(t) => t.with_timezone(&Local).format("%Y-%m-%d %H:%M").to_string(),
        Err(_) => rfc3339.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_drop_seconds_and_fractions() {
        let s = "2026-11-26T07:27:36.218826123+00:00";
        let t = DateTime::parse_from_rfc3339(s)
            .unwrap()
            .with_timezone(&Local);
        assert_eq!(local_stamp(s), t.format("%Y-%m-%d %H:%M").to_string());
        assert_eq!(local_stamp(s).len(), 16);
        assert_eq!(local_stamp("not a time"), "not a time");
        let u: DateTime<Utc> = DateTime::parse_from_rfc3339(s).unwrap().into();
        assert_eq!(local_clock(&u), t.format("%H:%M").to_string());
    }
}
