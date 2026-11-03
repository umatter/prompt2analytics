# Branch Protection Rules

This document describes the recommended branch protection settings for the prompt2analytics repository.

## Protected Branches

### `main` Branch

The `main` branch should have the following protection rules enabled:

#### Required Status Checks

A required check is named by its context: the `name:` of a job in a workflow that runs on every pull
request. Only `.github/workflows/ci.yml` does, and its four jobs report these contexts:

| Context | Job key in `ci.yml` | What it runs |
|---------|---------------------|--------------|
| `Format` | `fmt` | `cargo fmt --all -- --check` |
| `Test` | `test` | Unit and integration tests of p2a-core, p2a-mcp and p2a-cli under the CI feature sets, the p2a-core and p2a-mcp doctests, and the validation harness checks |
| `Clippy` | `clippy` | Clippy on p2a-core, p2a-mcp, p2a-cli, p2a-chat and p2a-validation under the same feature sets |
| `Dioxus (web) check` | `dioxus-check` | `cargo check` of the web app for `wasm32-unknown-unknown`, plus its host tests |

All four must pass before merging. When a job is added to `ci.yml` or renamed, update this table and the
snippet below in the same pull request; a required context that no job reports blocks every merge.

These workflows cannot be required, because they do not report on every pull request:

- **Security audit** (`audit.yml`, context `cargo audit`) runs weekly, on demand, and on pull requests
  that change dependency files. A pull request that changes none of them never gets the check.
- **Bundle Desktop App** (`desktop-bundle.yml`) and **Backend Image** (`backend-image.yml`) run on `v*`
  tags; **Deploy Web App (GitHub Pages)** (`web-deploy.yml`) runs on path-filtered pushes to `main` and on
  demand. Release builds are therefore not checked on pull requests.

No workflow measures code coverage, so there is no coverage check.

#### Review Requirements

- **Require pull request reviews**: At least 1 approval required
- **Dismiss stale reviews**: Dismiss approvals when new commits are pushed
- **Require review from code owners**: If CODEOWNERS file exists

#### Merge Requirements

- **Require branches to be up to date**: Branch must be current with `main`
- **Require status checks to pass**: Strict mode (all checks must pass)
- **Require linear history**: Optional, enforces rebase/squash workflow

#### Additional Protections

- **Do not allow bypassing the above settings**: Even for admins
- **Restrict who can push**: Only through pull requests
- **Do not allow force pushes**: Prevents history rewriting
- **Do not allow deletions**: Protects the main branch

## Why Each Check Matters

### Format
`cargo fmt --all -- --check` keeps one code style across the workspace and keeps formatting out of reviews.

### Test
`cargo test --tests` for p2a-core (`--no-default-features --features ci`), p2a-mcp
(`--no-default-features --features "http,llm,auth,websocket"`) and p2a-cli
(`--no-default-features --features spectral-analysis`), then the doctests of p2a-core and p2a-mcp, then the
validation harness: the p2a-validation tests, the methods-registry lint and the validation runner's own
tests. It catches logic errors, regressions, results that drift from their R references, and doc examples
that no longer compile. Benchmark-only tests are `#[ignore]`d and show as ignored. Release builds, `cuda`,
the `db` feature of p2a-mcp and the desktop app are not built; `docs/guides/TESTING.md` lists what each
feature set compiles.

### Clippy
Runs Clippy under the same feature sets as Test. Warnings are reported, not fatal: p2a-core carries a
backlog of existing warnings. The job fails when the code does not compile under those feature sets,
which is why it is required.

### Dioxus (web) check
Checks the web app for `wasm32-unknown-unknown`, the target that is deployed, and runs its host tests. It
catches a wasm-only break before the deploy workflow meets it on `main`.

## Setting Up Branch Protection

### Via GitHub Web UI

1. Go to **Settings** → **Branches** → **Add branch protection rule**
2. Branch name pattern: `main`
3. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (set to 1)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Status checks: `Format`, `Test`, `Clippy`, `Dioxus (web) check` (each from GitHub Actions)
   - ✅ Do not allow bypassing the above settings
4. Click **Create** or **Save changes**

### Via GitHub CLI

The endpoint takes a JSON body with all four top-level fields, so send it with `--input`. `15368` is
the GitHub Actions app, which reports the CI checks; pinning it stops another app from satisfying a
check by posting a status with the same name.

```bash
gh api --method PUT repos/{owner}/{repo}/branches/main/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "Format", "app_id": 15368},
      {"context": "Test", "app_id": 15368},
      {"context": "Clippy", "app_id": 15368},
      {"context": "Dioxus (web) check", "app_id": 15368}
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null
}
EOF
```

To confirm the contexts, list what CI reported on a pull request's head commit, and what the rule
requires:

```bash
gh api "repos/{owner}/{repo}/commits/<sha>/check-runs" --jq '.check_runs[] | [.name, .app.id] | @tsv'
gh api repos/{owner}/{repo}/branches/main/protection --jq '.required_status_checks.checks[].context'
```

## Feature Branches

Feature branches (e.g., `full-rust-migration`) can optionally have the same protections, or lighter rules depending on development needs.

## Emergency Procedures

If a critical fix is needed and CI is failing:

1. **Preferred**: Fix the CI issue first, then merge
2. **If urgent**: A repository admin can temporarily disable protection (document reason in commit message)
3. **Post-merge**: Re-enable protection immediately

Never leave branch protection disabled longer than necessary.
