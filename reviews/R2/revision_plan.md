# Revision Plan - Round 1 → Round 2

## Overview

Based on reviewer feedback from Round 1, this plan addresses 3 critical issues, 4 major issues, and 5 minor issues.

Estimated scope:
- LaTeX edits: 8 sections (abstract, introduction, methods, examples, comparison, performance, local, conclusion)
- Scripts to run: None required (no new analysis needed)
- New analyses: 0
- New figures/tables: 1 (summary limitation table for main text)

## Detailed Action Plan

---

### 1. Reframe Conceptual Contribution (CRITICAL)
**Source:** Reviewer 2, Major Comment #1
**Action Type:** tex_edit

**Current State:**
> "This paper describes chat-first data analytics: an application of established tool-augmented language model techniques..."

The paper uses "paradigm" language and frames "chat-first" as novel when reviewers see it as an implementation.

**Planned Change:**
1. In abstract: Change "approach" framing to emphasize software contribution
2. In Section 1.3: Add explicit acknowledgment that this applies existing techniques
3. In Section 2.1: Moderate "design principle" language
4. Throughout: Replace "paradigm" with "approach" or "implementation"

**Files Affected:**
- `paper/article-jss.tex`: Abstract edit
- `paper/sections/introduction.tex`: Section 1.3 edit
- `paper/sections/methods.tex`: Section 2.1, 2.2 edits

**Response Letter Text:**
> We thank the reviewer for this important clarification. We have reframed the contribution to emphasize the software implementation rather than claiming conceptual novelty. We now explicitly acknowledge that chat-first analytics applies established tool-augmented LLM techniques (Schick et al. 2023, Yao et al. 2023) to the econometrics domain. The revised abstract and introduction focus on the Rust library and validation as the primary contributions.

---

### 2. Move Limitation Findings to Main Text (CRITICAL)
**Source:** Reviewer 2, Major Comment #2; Reviewer 3, Major Comment #1
**Action Type:** tex_edit

**Current State:**
Key findings (47-60% multi-turn accuracy, 41-59% parameter F1) appear only in Section 7.3 and Appendix B.

**Planned Change:**
1. Add new subsection 2.4 "Practical accuracy expectations" with summary of key limitations
2. Create condensed Table showing metrics from Table 6 in main text
3. Add forward reference from Section 1.4 to set reader expectations early
4. Move key findings from Table B.1 into Section 7 proper

**Files Affected:**
- `paper/sections/methods.tex`: Add new subsection 2.4
- `paper/sections/introduction.tex`: Add forward reference
- `paper/sections/local.tex`: Restructure Section 7.3

**Response Letter Text:**
> We agree that key accuracy limitations were insufficiently prominent. We have added a new Section 2.4 "Practical accuracy expectations" that presents the multi-turn accuracy (47-60%), parameter extraction F1 (41-59%), and conversation completion rates (0-35%) early in the paper. This sets appropriate expectations before the polished examples in Section 4. We have also promoted the extended evaluation summary table (formerly Table B.1) to the main text.

---

### 3. Add Reproducibility Details for LLM Evaluation (CRITICAL)
**Source:** Reviewer 1, Major Comment #2; Reviewer 3, Major Comment #3
**Action Type:** tex_edit

**Current State:**
Missing: temperature settings, system prompts, exact model versions, random seeds.

**Planned Change:**
1. Add methodology paragraph to Section 7.1 specifying:
   - Temperature = 0 for all evaluations
   - System prompt (brief summary; full prompt in supplementary)
   - Model version dates/identifiers
   - Evaluation dates
2. Add supplementary file `llm_eval_prompts.txt` with exact prompts
3. Add reproducibility statement to Appendix A

**Files Affected:**
- `paper/sections/local.tex`: Section 7.1 methodology addition
- `paper/code/llm_eval/prompts/`: Document exact prompts used

**Response Letter Text:**
> We have added comprehensive reproducibility details to Section 7.1. All evaluations used temperature=0 for determinism. We specify exact model identifiers (e.g., "claude-3-5-haiku-20241022" not just "Claude 3.5 Haiku") and evaluation dates. The system prompt is summarized in the text with the complete prompt available in supplementary materials (llm_eval_prompts.txt). We acknowledge that API-served models may change over time and that exact replication may not be possible.

---

### 4. Add Failure Case to Section 4 Examples (MAJOR)
**Source:** Reviewer 2, Major Comment #3
**Action Type:** tex_edit

**Current State:**
Section 4.5 discusses "Realistic interaction patterns" with some failure examples, but these are brief. Section 4.1-4.4 show only successful interactions.

**Planned Change:**
Add new subsection 4.6 "A realistic multi-turn workflow" showing:
- Initial ambiguous request
- LLM misselects method
- User correction
- Parameter extraction failure
- Final successful completion after 3-4 turns

This demonstrates the realistic 2-3 correction cycles mentioned in Section 4.5.

**Files Affected:**
- `paper/sections/examples.tex`: Add new subsection 4.6

**Response Letter Text:**
> We have added Section 4.6 "A realistic multi-turn workflow" that demonstrates a complete analysis requiring multiple corrections. The example shows method misselection, parameter extraction failure, and eventual success—illustrating the 2-3 clarification exchanges typical of realistic use. This balances the polished single-turn examples in earlier subsections.

---

### 5. Clarify fixest Comparison (MAJOR)
**Source:** Reviewer 1, Major Comment #3
**Action Type:** tex_edit

**Current State:**
Section 5.5 mentions fixest in footnote 3 and Table 4, but no benchmark comparison.

**Planned Change:**
Add paragraph to Section 5.6 (Limitations) explaining:
- fixest was excluded from Table 5 benchmarks because its C++ core makes it fundamentally faster for HDFE
- fixest is the recommended choice for users prioritizing raw speed on HDFE problems
- prompt2analytics offers natural language interface as differentiating value

**Files Affected:**
- `paper/sections/comparison.tex`: Section 5.6 addition

**Response Letter Text:**
> We have added explicit discussion of fixest to Section 5.6. We clarify that fixest was excluded from Table 5 because its highly optimized C++ implementation makes direct speed comparison uninformative—fixest will outperform prompt2analytics on HDFE problems. Users prioritizing raw speed for high-dimensional fixed effects should use fixest directly. The differentiating value of prompt2analytics is the natural language interface and integration with other econometric methods, not speed leadership on any single method class.

---

### 6. Expand Method Coverage Discussion (MAJOR)
**Source:** Reviewer 2, Major Comment #4
**Action Type:** tex_edit

**Current State:**
Section 2.9 mentions "Bayesian methods and Heckman selection models are not implemented" briefly.

**Planned Change:**
Expand Section 2.9 to include:
- Table comparing implemented vs. unimplemented methods
- Explicit list: Bayesian inference, Heckman selection, Conley spatial SEs, SARIMA with auto-selection, quantile IV
- Recommendations for alternative tools for each missing category

**Files Affected:**
- `paper/sections/methods.tex`: Expand Section 2.9

**Response Letter Text:**
> We have expanded Section 2.9 to provide a more comprehensive discussion of method coverage limitations. We now include a table comparing implemented methods against key econometric techniques, with explicit acknowledgment of Bayesian inference, Heckman selection, Conley spatial standard errors, and other missing categories. For each gap, we recommend appropriate alternative tools (rstanarm, sampleSelection, etc.).

---

### 7. Moderate User Study Claims (MAJOR)
**Source:** Reviewer 2, Major Comment #3
**Action Type:** tex_edit

**Current State:**
Paper claims benefits of chat-first interfaces (accessibility, exploratory analysis) without user study evidence.

**Planned Change:**
1. Add limitation to Section 8.1: "The practical benefits of chat-first interfaces remain hypothesized; controlled user studies comparing task completion time and error rates against traditional interfaces are needed"
2. Moderate language throughout that implies demonstrated (vs. hypothesized) benefits
3. Change "enables" to "aims to enable" where appropriate

**Files Affected:**
- `paper/sections/conclusion.tex`: Section 8.1 limitation
- `paper/sections/introduction.tex`: Moderate language
- `paper/sections/methods.tex`: Moderate language

**Response Letter Text:**
> We acknowledge that user studies are essential for validating interface benefits and have added this as an explicit limitation in Section 8.1. We have moderated language throughout to distinguish between hypothesized benefits (accessibility, exploratory analysis) and demonstrated properties (numerical accuracy, performance). Conducting a rigorous user study was beyond the scope of this revision but is prioritized for future work.

---

### 8-12. Minor Issues

**8. CLI Syntax Note (Section 4.2)**
- Add footnote noting shell compatibility considerations

**9. Model Recommendations by Use Case (Section 7.5)**
- Expand Table 7 or add prose distinguishing single-turn vs. multi-turn recommendations

**10. Speedup Interpretation (Section 5.2)**
- Add sentence clarifying that small-n speedups reflect overhead, not computational differences (already partially present, strengthen)

**11. SurrealDB Justification (Section 3.4)**
- Add sentence: "SurrealDB was chosen for its embedded operation (no external service required) and document-flexible schema that accommodates evolving conversation structures"

**12. Typos and Rendering**
- Fix p.21 citation rendering
- Fix p.64 math rendering in warning message
- Review and fix other rendering issues identified

---

## Execution Order

1. [ ] Create backup of current paper source
2. [ ] Edit `paper/sections/methods.tex` (Issues 1, 2, 6, 7)
3. [ ] Edit `paper/sections/introduction.tex` (Issues 1, 2, 7)
4. [ ] Edit `paper/sections/examples.tex` (Issues 4, 8)
5. [ ] Edit `paper/sections/comparison.tex` (Issues 5, 10)
6. [ ] Edit `paper/sections/local.tex` (Issues 2, 3, 9)
7. [ ] Edit `paper/sections/conclusion.tex` (Issue 7)
8. [ ] Edit `paper/article-jss.tex` (Abstract - Issue 1)
9. [ ] Fix typos and rendering issues (Issue 12)
10. [ ] Build PDF and verify compilation
11. [ ] Write response letter

## Risk Assessment

**Potential Complications:**
- Reframing contribution may require careful balance to maintain paper coherence
- New Section 4.6 failure example needs to be realistic but not discouraging
- Adding limitation table to main text increases length

**Judgment Calls:**
- R2 wants "substantial reframing" vs R1/R3 satisfied with current approach
  - **Plan:** Make targeted language changes rather than complete rewrite; address core concern (overclaimed novelty) while preserving structure
- User study feasibility
  - **Plan:** Cannot conduct in revision timeframe; acknowledge limitation prominently instead
