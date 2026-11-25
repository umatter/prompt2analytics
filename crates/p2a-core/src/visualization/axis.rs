//! The x axis of the line and scatter charts: numbers, or days since 1970-01-01 labelled as dates.

use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeDelta};
use plotters::coord::{
    ranged1d::{KeyPointHint, NoDefaultFormatting, Ranged, ValueFormatter},
    types::RangedCoordf64,
};
use std::ops::Range;

/// Date labels that fit the default image width.
const MAX_DATE_LABELS: usize = 7;

/// An axis of numbers, which plotters labels, or of days since 1970-01-01.
pub(super) struct XAxis {
    coord: RangedCoordf64,
    dates: bool,
}

impl XAxis {
    pub(super) fn new(range: Range<f64>, dates: bool) -> Self {
        Self {
            coord: range.into(),
            dates,
        }
    }
}

/// The time `days` after 1970-01-01, to the second.
fn moment(days: f64) -> Option<NaiveDateTime> {
    let seconds = (days * 86_400.0).round();
    if seconds.abs() > 1e13 {
        return None;
    }
    TimeDelta::try_seconds(seconds as i64)
        .and_then(|d| NaiveDateTime::UNIX_EPOCH.checked_add_signed(d))
}

fn days_of(date: NaiveDate) -> f64 {
    date.signed_duration_since(NaiveDate::default()).num_days() as f64
}

/// At most `max` round dates in `range`: the finest of minutes, hours, days, months or years.
fn date_ticks(range: &Range<f64>, max: usize) -> Vec<f64> {
    let span = range.end - range.start;
    if max == 0 || !span.is_finite() || span <= 0.0 {
        return Vec::new();
    }
    const MINUTE: f64 = 1.0 / 1440.0;
    const HOUR: f64 = 1.0 / 24.0;
    let steps = [
        MINUTE,
        5.0 * MINUTE,
        15.0 * MINUTE,
        30.0 * MINUTE,
        HOUR,
        3.0 * HOUR,
        6.0 * HOUR,
        12.0 * HOUR,
        1.0,
        2.0,
        7.0,
        14.0,
    ];
    if let Some(step) = steps.iter().find(|s| (span / **s) < max as f64) {
        let first = (range.start / step).ceil();
        let count = ((range.end / step).floor() - first) as i64;
        return (0..=count).map(|i| (first + i as f64) * step).collect();
    }
    let (Some(start), Some(end)) = (moment(range.start), moment(range.end)) else {
        return Vec::new();
    };
    let months = |d: NaiveDateTime| i64::from(d.year()) * 12 + i64::from(d.month0());
    let (a, b) = (months(start), months(end));
    let mut step = 1i64;
    for s in [
        1, 3, 6, 12, 24, 60, 120, 240, 600, 1200, 6000, 12_000, 120_000,
    ] {
        step = s;
        // Counted from year 0, so that steps of whole years fall on round years.
        if (b.div_euclid(s) - a.div_euclid(s) + 1) as usize <= max {
            break;
        }
    }
    (a.div_euclid(step)..=b.div_euclid(step))
        .take(max + 1)
        .filter_map(|i| {
            let m = i * step;
            NaiveDate::from_ymd_opt(m.div_euclid(12) as i32, m.rem_euclid(12) as u32 + 1, 1)
        })
        .map(days_of)
        .filter(|d| *d >= range.start && *d <= range.end)
        .collect()
}

/// A tick's label, as short as the span of the axis allows.
fn date_label(days: f64, span: f64) -> String {
    let Some(t) = moment(days) else {
        return String::new();
    };
    let format = if span < 3.0 {
        "%m-%d %H:%M"
    } else if span <= 150.0 {
        "%Y-%m-%d"
    } else if span > 1096.0 && t.month() == 1 {
        "%Y"
    } else {
        "%Y-%m"
    };
    t.format(format).to_string()
}

impl Ranged for XAxis {
    type FormatOption = NoDefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        self.coord.map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        if !self.dates {
            return self.coord.key_points(hint);
        }
        let labels = !hint.weight().allow_light_points();
        let max = hint.max_num_points();
        date_ticks(
            &self.coord.range(),
            if labels {
                max.min(MAX_DATE_LABELS)
            } else {
                max
            },
        )
    }

    fn range(&self) -> Range<f64> {
        self.coord.range()
    }
}

impl ValueFormatter<f64> for XAxis {
    fn format_ext(&self, value: &f64) -> String {
        if self.dates {
            let range = self.coord.range();
            date_label(*value, range.end - range.start)
        } else {
            self.coord.format_ext(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(y: i32, m: u32, d: u32) -> f64 {
        days_of(NaiveDate::from_ymd_opt(y, m, d).unwrap())
    }

    fn labels(range: Range<f64>) -> Vec<String> {
        let span = range.end - range.start;
        date_ticks(&range, MAX_DATE_LABELS)
            .iter()
            .map(|d| date_label(*d, span))
            .collect()
    }

    #[test]
    fn date_labels_follow_the_span() {
        assert_eq!(
            labels(day(1948, 6, 1)..day(1962, 5, 1)),
            ["1950", "1955", "1960"]
        );
        assert_eq!(
            labels(day(2020, 1, 10)..day(2020, 12, 1)),
            ["2020-04", "2020-07", "2020-10"]
        );
        assert_eq!(
            labels(day(2020, 1, 1)..day(2020, 1, 6)),
            (1..=6).map(|d| format!("2020-01-0{d}")).collect::<Vec<_>>()
        );
        assert_eq!(labels(0.0..0.25).first().unwrap(), "01-01 00:00");
    }

    #[test]
    fn ticks_are_bounded_on_any_range() {
        for range in [0.0..1e-9, -1e8..1e8, 0.0..1e300, 5.0..5.0, -5e6..5e6] {
            for max in [0, 1, 7, 110] {
                assert!(date_ticks(&range, max).len() <= max + 1, "{range:?} {max}");
            }
        }
    }
}
