# Branch Protection Rules

This document describes the recommended branch protection settings for the prompt2analytics repository.

## Protected Branches

### `main` Branch

The `main` branch should have the following protection rules enabled:

#### Required Status Checks

All of these CI checks must pass before merging:

| Check | Job Name | Purpose |
|-------|----------|---------|
| **Test** | `test` | Runs full test suite for all crates |
| **Clippy** | `clippy` | Enforces Rust best practices and catches common mistakes |
| **Format** | `format` | Ensures consistent code formatting |
| **Build** | `build` | Verifies debug and release builds succeed |
| **Documentation** | `docs` | Ensures doc comments compile without warnings |

**Optional (informational):**
- **Coverage** (`coverage`) - Code coverage reporting via Codecov

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

### Test (`cargo test --workspace`)
Catches logic errors and regressions. Tests run for p2a-core, p2a-cli, p2a-mcp, and p2a-dioxus.

### Clippy (`cargo clippy -D warnings`)
Enforces Rust idioms, catches potential bugs, and prevents common performance issues. Runs with strict warnings-as-errors.

### Format (`cargo fmt --check`)
Maintains consistent code style across the codebase. Prevents formatting debates in PRs.

### Build
Verifies both debug and release builds work. Release builds catch optimization-related issues.

### Documentation
Ensures all public API documentation compiles. Catches broken links and invalid examples.

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
   - ✅ Status checks: `test`, `clippy`, `format`, `build`, `docs`
   - ✅ Do not allow bypassing the above settings
4. Click **Create** or **Save changes**

### Via GitHub CLI

```bash
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  -F required_status_checks='{"strict":true,"contexts":["test","clippy","format","build","docs"]}' \
  -F enforce_admins=true \
  -F required_pull_request_reviews='{"dismiss_stale_reviews":true,"required_approving_review_count":1}'
```

## Feature Branches

Feature branches (e.g., `full-rust-migration`) can optionally have the same protections, or lighter rules depending on development needs.

## Emergency Procedures

If a critical fix is needed and CI is failing:

1. **Preferred**: Fix the CI issue first, then merge
2. **If urgent**: A repository admin can temporarily disable protection (document reason in commit message)
3. **Post-merge**: Re-enable protection immediately

Never leave branch protection disabled longer than necessary.
