//! Dates in user columns: reading them for a chart axis, and continuing a regular sequence.

use chrono::{Datelike, Days, Months, NaiveDate};
use polars::prelude::*;

/// The date of an unambiguous text: `1949-01-31`, `1949/01/31`, `1949-01`, `1949/01`, `1949Q1`
/// or `1949-Q1` (a month or quarter is its first day). Day-first and month-first forms are
/// not read, because they cannot be told apart.
pub fn parse_date_text(text: &str) -> Option<NaiveDate> {
    let text = text.trim();
    if !text.is_ascii() || text.len() < 6 || text.len() > 10 {
        return None;
    }
    let number = |s: &str, digits: std::ops::RangeInclusive<usize>| {
        (digits.contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse::<u32>().ok())
            .flatten()
    };
    let year = number(text.get(..4)?, 4..=4)? as i32;
    let rest = &text[4..];
    let quarter = rest.trim_start_matches(['-', ' ']);
    if let Some(q) = quarter.strip_prefix(['Q', 'q']) {
        let q = number(q, 1..=1).filter(|q| (1..=4).contains(q))?;
        return NaiveDate::from_ymd_opt(year, q * 3 - 2, 1);
    }
    let sep = rest.chars().next().filter(|c| matches!(c, '-' | '/'))?;
    let mut parts = rest[1..].split(sep);
    let month = number(parts.next()?, 1..=2)?;
    let day = match parts.next() {
        Some(d) => number(d, 1..=2)?,
        None => 1,
    };
    if parts.next().is_some() {
        return None;
    }
    NaiveDate::from_ymd_opt(year, month, day)
}

fn days_of(date: NaiveDate) -> i32 {
    date.signed_duration_since(NaiveDate::default()).num_days() as i32
}

/// A time column as days since 1970-01-01: a `Date` column itself, a `Datetime` column as
/// `Float64` days, a text column whose every value reads as a date ([`parse_date_text`]) as
/// `Date`. None for any other column; empty text is missing.
pub fn as_date_column(column: &Column) -> Option<Column> {
    match column.dtype() {
        DataType::Date => Some(column.clone()),
        DataType::Datetime(unit, _) => {
            let per_day = match unit {
                TimeUnit::Milliseconds => 86_400e3,
                TimeUnit::Microseconds => 86_400e6,
                TimeUnit::Nanoseconds => 86_400e9,
            };
            let physical = column.to_physical_repr();
            let days: Float64Chunked = physical
                .i64()
                .ok()?
                .iter()
                .map(|v| v.map(|v| v as f64 / per_day))
                .collect();
            Some(days.with_name(column.name().clone()).into_column())
        }
        DataType::String => {
            let mut days: Vec<Option<i32>> = Vec::with_capacity(column.len());
            for text in column.str().ok()?.iter() {
                days.push(match text.map(str::trim) {
                    None | Some("") => None,
                    Some(t) => Some(days_of(parse_date_text(t)?)),
                });
            }
            days.iter().any(Option::is_some).then_some(())?;
            Series::new(column.name().clone(), days)
                .cast(&DataType::Date)
                .ok()
                .map(Series::into_column)
        }
        _ => None,
    }
}

/// The common value of `diffs`, if they are all equal and positive.
fn constant(mut diffs: impl Iterator<Item = i64>) -> Option<u64> {
    let first = diffs.next().filter(|d| *d > 0)?;
    diffs.all(|d| d == first).then_some(first as u64)
}

/// The dates after `dates` when they follow each other regularly: a whole number of months on
/// one day of the month (or on month ends), else a whole number of days. At least three dates.
fn continue_dates(dates: &[NaiveDate], steps: usize) -> Option<Vec<NaiveDate>> {
    if dates.len() < 3 {
        return None;
    }
    let month = |d: &NaiveDate| i64::from(d.year()) * 12 + i64::from(d.month0());
    let last = *dates.last()?;
    let month_end = |d: &NaiveDate| {
        d.checked_add_days(Days::new(1))
            .is_some_and(|n| n.day() == 1)
    };
    let ends = dates.iter().all(month_end);
    let same_day = dates[0].day() <= 28 && dates.iter().all(|d| d.day() == dates[0].day());
    if (ends || same_day)
        && let Some(k) = constant(dates.windows(2).map(|w| month(&w[1]) - month(&w[0])))
    {
        return (1..=steps)
            .map(|i| {
                let months = Months::new(u32::try_from(k.checked_mul(i as u64)?).ok()?);
                if ends {
                    // The first of the month after, less a day.
                    last.with_day(1)?
                        .checked_add_months(months)?
                        .checked_add_months(Months::new(1))?
                        .pred_opt()
                } else {
                    last.checked_add_months(months)
                }
            })
            .collect();
    }
    let days = dates.windows(2).map(|w| (w[1] - w[0]).num_days());
    let k = constant(days)?;
    (1..=steps)
        .map(|i| last.checked_add_days(Days::new(k.checked_mul(i as u64)?)))
        .collect()
}

/// The time column of a forecast, named `name`: the rows of `df` and `steps` more. Continues the
/// one regular date column of `df`, else consecutive whole numbers in a column called "year",
/// else counts 1, 2, ... (also when several date columns would do).
pub fn extended_time(df: &DataFrame, steps: usize, name: &str) -> Column {
    let n = df.height();
    let mut found: Vec<Vec<i32>> = Vec::new();
    for column in df.get_columns() {
        let Some(dates) = as_date_column(column).filter(|c| c.dtype() == &DataType::Date) else {
            continue;
        };
        let physical = dates.to_physical_repr();
        let Ok(days) = physical.i32() else { continue };
        let dates: Option<Vec<NaiveDate>> = days
            .iter()
            .map(|d| {
                NaiveDate::default().checked_add_signed(chrono::TimeDelta::days(i64::from(d?)))
            })
            .collect();
        if let Some(dates) = dates
            && let Some(next) = continue_dates(&dates, steps)
        {
            found.push(dates.into_iter().chain(next).map(days_of).collect());
        }
    }
    if let [days] = found.as_slice()
        && let Ok(column) = Series::new(name.into(), days.as_slice()).cast(&DataType::Date)
    {
        return column.into_column();
    }
    if found.is_empty()
        && let Some(year) = df
            .get_columns()
            .iter()
            .find(|c| c.name().eq_ignore_ascii_case("year") && c.dtype().is_integer())
        && let Ok(years) = year.cast(&DataType::Int64)
        && let Ok(years) = years.i64()
        && let Some(years) = years.iter().collect::<Option<Vec<i64>>>()
        && let Some(&last) = years.last()
        && years.windows(2).all(|w| w[0].checked_add(1) == Some(w[1]))
        && let Some(end) = last.checked_add(steps as i64)
    {
        let all: Vec<i64> = years.into_iter().chain(last + 1..=end).collect();
        return Column::new(name.into(), all);
    }
    let count: Vec<i64> = (1..=(n + steps) as i64).collect();
    Column::new(name.into(), count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> NaiveDate {
        NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn only_unambiguous_texts_are_dates() {
        for (text, date) in [
            ("1949-01", "1949-01-01"),
            (" 1949/2 ", "1949-02-01"),
            ("2024-02-29", "2024-02-29"),
            ("2024/3/9", "2024-03-09"),
            ("2001Q3", "2001-07-01"),
            ("2001-q4", "2001-10-01"),
            ("2001 Q1", "2001-01-01"),
        ] {
            assert_eq!(parse_date_text(text), Some(d(date)), "{text}");
        }
        for text in [
            "",
            "1949",
            "01/02/2020",
            "2020-13",
            "2023-02-29",
            "2020-01/02",
            "2001Q5",
            "2001-01-01-1",
            "2020-+1",
            "1949-01x",
            "１９４９-01",
            "194901",
            "2020-",
        ] {
            assert_eq!(parse_date_text(text), None, "{text}");
        }
    }

    #[test]
    fn text_and_datetime_columns_become_days() {
        let text = Column::new("m".into(), [Some("1970-01-03"), None, Some("")]);
        let days = as_date_column(&text).unwrap();
        assert_eq!(days.dtype(), &DataType::Date);
        assert_eq!(days.null_count(), 2);
        let mixed = Column::new("m".into(), ["1970-01", "Jan"]);
        assert!(as_date_column(&mixed).is_none());
        assert!(as_date_column(&Column::new("m".into(), [None::<&str>])).is_none());
        assert!(as_date_column(&Column::new("m".into(), [1i64])).is_none());
        let half = Column::new("t".into(), [43_200_000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();
        assert_eq!(
            as_date_column(&half).unwrap().f64().unwrap().get(0),
            Some(0.5)
        ); // na-ok: built here
    }

    #[test]
    fn regular_dates_continue() {
        let next = |v: &[&str], k| {
            continue_dates(&v.iter().map(|t| d(t)).collect::<Vec<_>>(), k)
                .map(|v| v.iter().map(|d| d.to_string()).collect::<Vec<_>>())
        };
        assert_eq!(
            next(&["1960-10-01", "1960-11-01", "1960-12-01"], 2).unwrap(),
            ["1961-01-01", "1961-02-01"]
        );
        assert_eq!(
            next(&["2023-11-30", "2023-12-31", "2024-01-31"], 2).unwrap(),
            ["2024-02-29", "2024-03-31"]
        );
        assert_eq!(
            next(&["2024-01-01", "2024-01-08", "2024-01-15"], 1).unwrap(),
            ["2024-01-22"]
        );
        assert!(next(&["2024-01-01", "2024-01-02", "2024-01-04"], 1).is_none());
        assert!(next(&["2024-01-03", "2024-01-02", "2024-01-01"], 1).is_none());
        assert!(next(&["2024-01-01", "2024-01-02"], 1).is_none());
    }

    #[test]
    fn forecast_time_falls_back_to_a_count() {
        let y = df!("year" => [2001i64, 2002], "v" => [1.0, 2.0]).unwrap();
        let t = extended_time(&y, 2, "time");
        assert_eq!(t.i64().unwrap().iter().flatten().last(), Some(2004));
        let gap = df!("year" => [2001i64, 2003], "v" => [1.0, 2.0]).unwrap();
        let t = extended_time(&gap, 1, "time");
        assert_eq!(
            t.i64().unwrap().iter().flatten().collect::<Vec<_>>(),
            [1, 2, 3]
        );
        let m = df!("m" => ["2001Q1", "2001Q2", "2001Q3"]).unwrap();
        let t = extended_time(&m, 2, "time");
        assert_eq!((t.len(), t.dtype()), (5, &DataType::Date));
        let two = df!("a" => ["2001Q1", "2001Q2", "2001Q3"], "b" => ["2001Q1", "2001Q2", "2001Q3"]);
        assert_eq!(
            extended_time(&two.unwrap(), 1, "time").dtype(),
            &DataType::Int64
        );
    }
}
