//! Golden files: committed expected text output (CLI exhibits).

use std::path::Path;

/// At most this many differing lines are shown in a mismatch message.
const MAX_SHOWN: usize = 20;

/// Compare `actual` with the committed golden file at `path`; line endings are
/// normalised to `\n`. With `P2A_UPDATE_GOLDEN=1` the file is (re)written instead and
/// the call passes: do that only after the numbers in the output were checked against R.
///
/// Panics with a line diff when the texts differ, or when the file does not exist.
pub fn assert_golden(path: &Path, actual: &str) {
    let update = std::env::var_os("P2A_UPDATE_GOLDEN").is_some_and(|v| v == "1");
    if let Err(message) = check_golden(path, actual, update) {
        panic!("{message}");
    }
}

fn normalise(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn check_golden(path: &Path, actual: &str, update: bool) -> Result<(), String> {
    let actual = normalise(actual);
    if update {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        return std::fs::write(path, &actual)
            .map_err(|e| format!("cannot write golden file {}: {e}", path.display()));
    }
    let expected = match std::fs::read_to_string(path) {
        Ok(text) => normalise(&text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "golden file {} does not exist; check the output against R, \
                 then create it with P2A_UPDATE_GOLDEN=1",
                path.display()
            ));
        }
        Err(e) => return Err(format!("cannot read golden file {}: {e}", path.display())),
    };
    if expected == actual {
        return Ok(());
    }
    let exp: Vec<&str> = expected.split('\n').collect();
    let act: Vec<&str> = actual.split('\n').collect();
    let differing: Vec<usize> = (0..exp.len().max(act.len()))
        .filter(|&i| exp.get(i) != act.get(i))
        .collect();
    let first = differing.first().map_or(1, |i| i + 1);
    let mut msg = format!("golden file {} differs at line {first}:\n", path.display());
    for &i in differing.iter().take(MAX_SHOWN) {
        msg.push_str(&format!("line {}:\n", i + 1));
        msg.push_str(&format!(
            "- {}\n",
            exp.get(i).copied().unwrap_or("[end of file]")
        ));
        msg.push_str(&format!(
            "+ {}\n",
            act.get(i).copied().unwrap_or("[end of file]")
        ));
    }
    if differing.len() > MAX_SHOWN {
        msg.push_str(&format!(
            "... and {} more differing lines\n",
            differing.len() - MAX_SHOWN
        ));
    }
    msg.push_str(&format!(
        "golden: {} lines, actual: {} lines\n",
        expected.lines().count(),
        actual.lines().count()
    ));
    msg.push_str(
        "if the change is intended and its numbers were checked against R, \
         rerun with P2A_UPDATE_GOLDEN=1",
    );
    Err(msg)
}

#[cfg(test)]
mod tests {
    use super::check_golden;

    #[test]
    fn update_compare_and_mismatch() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "p2a-validation-golden-{}-{}",
            std::process::id(),
            nanos
        ));
        let p = dir.join("nested").join("out.txt");

        let missing = check_golden(&p, "a\n", false).unwrap_err();
        assert!(missing.contains("does not exist"), "{missing}");
        assert!(missing.contains("P2A_UPDATE_GOLDEN=1"), "{missing}");

        check_golden(&p, "a\nb\n", true).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nb\n");
        check_golden(&p, "a\r\nb\r\n", false).unwrap();

        let err = check_golden(&p, "a\nc\n", false).unwrap_err();
        assert!(err.contains("differs at line 2"), "{err}");
        assert!(err.contains("- b"), "{err}");
        assert!(err.contains("+ c"), "{err}");
        assert!(err.contains("rerun with P2A_UPDATE_GOLDEN=1"), "{err}");

        let longer = check_golden(&p, "a\nb\nc\n", false).unwrap_err();
        assert!(longer.contains("differs at line 3"), "{longer}");
        assert!(
            longer.contains("golden: 2 lines, actual: 3 lines"),
            "{longer}"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
