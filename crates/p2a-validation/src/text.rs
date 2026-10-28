//! Parsers for the CLI's text output (`p2a reg ols`, `p2a panel ...`).

/// `^-{10,}\s*$`
fn is_dash_rule(line: &str) -> bool {
    let t = line.trim_end();
    t.len() >= 10 && t.bytes().all(|b| b == b'-')
}

/// `^Variable\s+.*\b[tz]-value\s+p-value\s*$`
fn is_table_header(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("Variable") else {
        return false;
    };
    if !rest.starts_with(char::is_whitespace) {
        return false;
    }
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let [.., stat, p] = tokens.as_slice() else {
        return false;
    };
    if *p != "p-value" {
        return false;
    }
    ["t-value", "z-value"].iter().any(|suffix| {
        stat.strip_suffix(suffix).is_some_and(|before| {
            before
                .chars()
                .last()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
        })
    })
}

fn parse_row(line: &str) -> Option<(String, Vec<f64>)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 5 {
        return None;
    }
    let split = tokens.len() - 4;
    let last = tokens[tokens.len() - 1];
    let stripped = last.trim_end_matches('*');
    let last = if stripped.len() < last.len() {
        stripped
    } else {
        last.strip_suffix('.').unwrap_or(last)
    };
    let mut numbers = Vec::with_capacity(4);
    for (j, tok) in tokens[split..].iter().enumerate() {
        let tok = if j == 3 { last } else { tok };
        numbers.push(tok.parse::<f64>().ok()?);
    }
    Some((tokens[..split].join(" "), numbers))
}

/// Rows of a CLI coefficient table: `(name, [coef, se, stat, p])`.
///
/// The table starts at the first line `Variable ... t-value p-value` (or `z-value`),
/// skips a dash rule right after it and ends at the next dash rule. The significance
/// stars after the p-value (or a single trailing `.`) are stripped; `NaN`, `inf` and
/// `-inf` are read. A name may contain spaces.
///
/// Panics with `no coefficient table found` if there is no header line, and with
/// ``cannot parse coefficient row `<line>` `` if a table line does not parse.
pub fn coefficient_rows(text: &str) -> Vec<(String, Vec<f64>)> {
    let mut lines = text.lines().skip_while(|l| !is_table_header(l));
    if lines.next().is_none() {
        panic!("no coefficient table found");
    }
    let mut lines = lines.peekable();
    if lines.peek().is_some_and(|l| is_dash_rule(l)) {
        lines.next();
    }
    lines
        .take_while(|l| !is_dash_rule(l))
        .map(|l| parse_row(l).unwrap_or_else(|| panic!("cannot parse coefficient row `{l}`")))
        .collect()
}

/// The number after `label` on the first line that starts with `label` (leading
/// whitespace ignored), e.g. `labelled(text, "R-squared:")`. Trailing `,`, `;` and `)`
/// are stripped from the number. `None` if no line starts with `label` or the first
/// such line has no number after it.
pub fn labelled(text: &str, label: &str) -> Option<f64> {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|l| l.starts_with(label))?;
    let token = line[label.len()..].split_whitespace().next()?;
    token.trim_end_matches([',', ';', ')']).parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rule_matches_t_and_z_tables_only() {
        assert!(is_table_header(
            "Variable         Coefficient   Std. Error    t-value    p-value"
        ));
        assert!(is_table_header("Variable  Coef  SE  z-value  p-value  "));
        assert!(!is_table_header("  Variable  Coef  SE  t-value  p-value"));
        assert!(!is_table_header("Variables  Coef  SE  t-value  p-value"));
        assert!(!is_table_header("Variable  Coef  SE  abct-value  p-value"));
        assert!(!is_table_header("Variable  Coef  SE  t-value"));
    }

    #[test]
    fn dash_rule_needs_ten_dashes() {
        assert!(is_dash_rule("----------"));
        assert!(is_dash_rule(
            "------------------------------------------------------------  "
        ));
        assert!(!is_dash_rule("---------"));
        assert!(!is_dash_rule("---------- x"));
    }

    #[test]
    fn row_reads_infinities_and_rejects_short_rows() {
        let (name, v) = parse_row("x  inf  -inf  NaN  1.0**").unwrap();
        assert_eq!(name, "x");
        assert_eq!(v[0], f64::INFINITY);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert!(v[2].is_nan());
        assert_eq!(v[3], 1.0);
        assert!(parse_row("x 1 2 3").is_none());
        assert!(parse_row("x 1 2 three 4").is_none());
    }

    #[test]
    #[should_panic(expected = "cannot parse coefficient row `oops`")]
    fn bad_row_panics_with_the_line() {
        let _ =
            coefficient_rows("Variable Coef SE t-value p-value\n----------\noops\n----------\n");
    }
}
