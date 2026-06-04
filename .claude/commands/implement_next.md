---
description: Implement the next highest-priority unimplemented method from the discovery queue
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, Task, Skill
---

# Implement Next Method

Automatically picks the next highest-priority unimplemented method(s) from the discovery queue and implements them.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `N` | integer | 1 | Number of methods to implement from the queue |

**Usage:**
```
/implement_next        # Implement 1 method (default)
/implement_next 3      # Implement next 3 methods
/implement_next 5      # Implement next 5 methods
```

**Notes:**
- Methods are processed sequentially in priority order
- If a method fails/blocks, it's marked as blocked and the next method is attempted
- A batch summary is shown at the end when N > 1
- Implementation stops early if fewer than N pending methods remain

## Workflow

### Step 1: Load the Implementation Queue

1. **Read the queue file**: `docs/discovery/implementation_queue.json`

   If file doesn't exist:
   ```
   ⚠️ No implementation queue found.

   Run /discover_methods <index-url> first to generate a prioritized queue.
   Example: /discover_methods https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html
   ```
   **STOP** - Do not proceed.

2. **Parse the queue** to find methods with `status: "pending"`

### Step 2: Select Methods to Implement

1. **Parse the N argument** (default: 1)
   - If user provided an argument, parse as integer
   - Validate N >= 1

2. **Find all pending methods** where `status == "pending"` (list is pre-sorted by priority)

3. **Select up to N methods** from the pending list

4. If no pending methods:
   ```
   ✅ ALL METHODS IMPLEMENTED

   All [total] methods in the queue have been implemented.

   To add more methods:
   - Run /discover_methods on a new package index
   - Or manually add entries to docs/discovery/implementation_queue.json
   ```
   **STOP** - Do not proceed.

5. **Display the batch plan** (when N > 1):
   ```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📋 BATCH IMPLEMENTATION PLAN
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Implementing [actual_count] methods (requested: [N]):

   │ # │ Method          │ Category          │ Score │ Complexity │
   │───│─────────────────│───────────────────│───────│────────────│
   │ 1 │ var.test        │ Hypothesis        │ 95    │ Simple     │
   │ 2 │ prop.test       │ Hypothesis        │ 92    │ Simple     │
   │ 3 │ binom.test      │ Hypothesis        │ 90    │ Simple     │

   Current progress: [completed]/[total] methods done ([percentage]%)
   After this batch: [new_completed]/[total] ([new_percentage]%)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ```

6. **Display single method info** (when N == 1):
   ```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🎯 NEXT METHOD TO IMPLEMENT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Method:      [method_name]
   Category:    [category]
   Priority:    [score]/100
   Complexity:  [Simple/Medium/Complex]
   Source:      [source_package]
   Doc URL:     [doc_url]

   Progress: [completed]/[total] methods done ([percentage]%)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ```

### Step 3: Update Status to "in_progress"

1. Update the queue file:
   ```json
   {
     "method": "t.test",
     "status": "in_progress",  // Changed from "pending"
     "started_at": "2026-01-18T12:00:00Z"
   }
   ```

2. Save the updated queue file

### Step 4: Run Implementation

1. **Invoke the implement_metrics skill** with the method's doc URL:
   ```
   /implement_metrics [doc_url]
   ```

2. The implement_metrics command will:
   - Check for existing implementations (Phase 0)
   - Research the method (Phase 1)
   - Plan the implementation (Phase 2)
   - Implement with MCP tool (Phase 3)
   - Validate against R (Phase 4)
   - Document (Phase 5)
   - Create benchmarks (Phase 6)
   - **Execute tests & benchmarks, record performance results (Phase 7)**

3. **Phase 7 is MANDATORY** and includes:
   - Running `cargo test -p p2a-core -- test_validate_[method]`
   - Running `cargo bench -p p2a-core -- [method]`
   - Running R benchmark script
   - Updating validation document with actual performance numbers

### Step 5: Update Status on Completion

After implement_metrics completes successfully:

1. **Update the queue file**:
   ```json
   {
     "method": "t.test",
     "status": "completed",  // Changed from "in_progress"
     "started_at": "2026-01-18T12:00:00Z",
     "completed_at": "2026-01-18T14:30:00Z",
     "rust_function": "run_t_test",
     "mcp_tool": "t_test",
     "validation": "passed"
   }
   ```

2. **Update the discovery report** (`docs/discovery/[package]-[date].md`):
   - Change the method's row from ❌ to ✅
   - Add the implementation location

3. **Display completion summary**:
   ```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✅ METHOD IMPLEMENTED: [method_name]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Implementation: crates/p2a-core/src/[path].rs
   MCP Tool:       [tool_name]
   Validation:     ✅ Passed (matches R within tolerance)
   Performance:    ✅ Benchmarked (Rust Xms vs R Yms = ~Zx speedup)

   Artifacts:
   - Validation doc: validation/[category]/[method].md
   - R benchmark:    ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R

   Progress: [completed]/[total] methods done ([percentage]%)

   NEXT UP: [next_method_name] ([next_priority]/100)

   Run /implement_next again to continue, or /queue_status to see progress.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ```

### Step 6: Batch Loop (when N > 1)

When implementing multiple methods:

1. **For each method in the batch** (i = 1 to N):
   - Display progress: `[i/N] Implementing: [method_name]`
   - Update status to "in_progress"
   - Run `/implement_metrics [doc_url]`
   - On success: Update status to "completed", record result
   - On failure: Update status to "blocked", record reason, continue to next

2. **Track batch results**:
   ```
   batch_results = {
     "requested": N,
     "completed": [...],
     "blocked": [...],
     "skipped": [...]  // if fewer pending than N
   }
   ```

3. **After all methods processed**, display batch summary (Step 8)

### Step 7: Handle Failures

If implementation fails or is blocked:

1. **Update status to "blocked"**:
   ```json
   {
     "method": "t.test",
     "status": "blocked",
     "started_at": "2026-01-18T12:00:00Z",
     "blocked_at": "2026-01-18T13:00:00Z",
     "blocked_reason": "Missing dependency: special function library"
   }
   ```

2. **Skip to next method** and inform user:
   ```
   ⚠️ METHOD BLOCKED: [method_name]

   Reason: [blocked_reason]

   Skipping to next method. You can revisit blocked methods later with:
   /implement_metrics [doc_url]
   ```

### Step 8: Batch Summary (when N > 1)

After processing all methods in the batch:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BATCH IMPLEMENTATION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requested: [N] methods
Completed: [completed_count]
Blocked:   [blocked_count]
Skipped:   [skipped_count] (not enough pending methods)

COMPLETED METHODS:
│ # │ Method          │ Rust Function    │ MCP Tool     │ Validation │
│───│─────────────────│──────────────────│──────────────│────────────│
│ 1 │ var.test        │ var_test         │ var_test     │ ✅ Passed  │
│ 2 │ prop.test       │ prop_test        │ prop_test    │ ✅ Passed  │
│ 3 │ binom.test      │ binom_test       │ binom_test   │ ✅ Passed  │

BLOCKED METHODS (if any):
│ Method          │ Reason                                    │
│─────────────────│───────────────────────────────────────────│
│ complex.method  │ Missing dependency: special function lib  │

Overall Progress: [total_completed]/[total] methods done ([percentage]%)

NEXT UP: [next_method_name] ([next_priority]/100)

Run /implement_next [N] to continue with more methods.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Queue File Format

The implementation queue is stored at `docs/discovery/implementation_queue.json`:

```json
{
  "source": "R stats package",
  "source_url": "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html",
  "generated_at": "2026-01-18T10:00:00Z",
  "total_methods": 20,
  "completed": 3,
  "in_progress": 1,
  "blocked": 0,
  "pending": 16,
  "methods": [
    {
      "rank": 1,
      "method": "t.test",
      "category": "Hypothesis Testing",
      "priority_score": 92,
      "complexity": "Simple",
      "doc_url": "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html",
      "status": "completed",
      "started_at": "2026-01-18T11:00:00Z",
      "completed_at": "2026-01-18T12:30:00Z",
      "rust_function": "run_t_test",
      "mcp_tool": "t_test"
    },
    {
      "rank": 2,
      "method": "chisq.test",
      "category": "Hypothesis Testing",
      "priority_score": 88,
      "complexity": "Simple",
      "doc_url": "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html",
      "status": "pending"
    }
  ]
}
```

---

## Related Commands

- `/discover_methods <url>` - Generate a new implementation queue from a package index
- `/queue_status` - View current queue progress without implementing
- `/implement_metrics <url>` - Implement a specific method (bypasses queue)

---

## Example Sessions

### Single Method (Default)

```
User: /implement_next

Claude:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 NEXT METHOD TO IMPLEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method:      var.test
Category:    Hypothesis Testing
Priority:    95/100
Complexity:  Simple
Source:      R stats
Doc URL:     https://stat.ethz.ch/R-manual/R-devel/library/stats/html/var.test.html

Progress: 0/20 methods done (0%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting implementation...

[... implement_metrics workflow runs ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ METHOD IMPLEMENTED: var.test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implementation: crates/p2a-core/src/stats/vartest.rs
MCP Tool:       var_test
Validation:     ✅ Passed

Progress: 1/20 methods done (5%)

NEXT UP: prop.test (92/100)

Run /implement_next again to continue.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Batch Mode (Multiple Methods)

```
User: /implement_next 3

Claude:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 BATCH IMPLEMENTATION PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implementing 3 methods:

│ # │ Method          │ Category          │ Score │ Complexity │
│───│─────────────────│───────────────────│───────│────────────│
│ 1 │ var.test        │ Hypothesis        │ 95    │ Simple     │
│ 2 │ prop.test       │ Hypothesis        │ 92    │ Simple     │
│ 3 │ binom.test      │ Hypothesis        │ 90    │ Simple     │

Current progress: 0/20 methods done (0%)
After this batch: 3/20 (15%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/3] Implementing: var.test
[... implement_metrics workflow runs ...]
✅ var.test completed

[2/3] Implementing: prop.test
[... implement_metrics workflow runs ...]
✅ prop.test completed

[3/3] Implementing: binom.test
[... implement_metrics workflow runs ...]
✅ binom.test completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BATCH IMPLEMENTATION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requested: 3 methods
Completed: 3
Blocked:   0
Skipped:   0

COMPLETED METHODS:
│ # │ Method          │ Rust Function    │ MCP Tool     │ Validation │
│───│─────────────────│──────────────────│──────────────│────────────│
│ 1 │ var.test        │ var_test         │ var_test     │ ✅ Passed  │
│ 2 │ prop.test       │ prop_test        │ prop_test    │ ✅ Passed  │
│ 3 │ binom.test      │ binom_test       │ binom_test   │ ✅ Passed  │

Overall Progress: 3/20 methods done (15%)

NEXT UP: fligner.test (88/100)

Run /implement_next 3 to continue with more methods.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Batch with Blocked Method

```
User: /implement_next 3

Claude:
[... batch starts ...]

[1/3] Implementing: complex.method
⚠️ BLOCKED: Missing dependency

[2/3] Implementing: simple.method
✅ simple.method completed

[3/3] Implementing: another.method
✅ another.method completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BATCH IMPLEMENTATION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requested: 3 methods
Completed: 2
Blocked:   1
Skipped:   0

COMPLETED METHODS:
│ # │ Method          │ Rust Function    │ MCP Tool     │ Validation │
│───│─────────────────│──────────────────│──────────────│────────────│
│ 1 │ simple.method   │ simple_method    │ simple       │ ✅ Passed  │
│ 2 │ another.method  │ another_method   │ another      │ ✅ Passed  │

BLOCKED METHODS:
│ Method          │ Reason                              │
│─────────────────│─────────────────────────────────────│
│ complex.method  │ Missing dependency: special fn lib  │

Overall Progress: 5/20 methods done (25%)

Run /implement_next to continue.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
