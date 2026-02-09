# Review: Chat-First Data Analytics

**Reviewer 1**
**Expertise:** Statistical Computing, R Package Development, Software Engineering
**Recommendation:** Minor Revision

---

## Summary

This paper introduces "chat-first data analytics," an approach that uses LLMs for natural language interpretation while delegating computation to pre-validated statistical implementations. The software package **prompt2analytics** implements this paradigm through a Rust-based econometrics library exposed via the Model Context Protocol (MCP). The paper is well-written, technically sound, and makes a useful contribution to the field of statistical software.

---

## Major Comments

### 1. Software Availability and Installation Documentation

The paper references a GitHub repository but provides incomplete installation instructions for all platforms. While Appendix D covers deployment, the main text should include a brief summary of the installation process and system requirements for the typical JSS reader who wants to reproduce the examples.

**Suggestion:** Add a brief installation section early in the paper (perhaps at the end of Section 1.6) with commands for the most common case (Linux with cargo installed).

### 2. Reproducibility of LLM Evaluation Results

The LLM evaluation in Section 7 is valuable but raises reproducibility concerns. LLM outputs are non-deterministic, and the evaluated models may be updated or deprecated over time. The paper acknowledges this partially but should more explicitly address:

- What temperature settings were used?
- Were system prompts held constant across evaluations?
- How should readers interpret results when they cannot reproduce them with the same model versions?

**Suggestion:** Add a reproducibility statement specifically for the LLM evaluation, and consider including the exact prompts used in supplementary materials.

### 3. Comparison with fixest

Table 4 and the benchmark discussion acknowledge that **fixest** outperforms prompt2analytics for HDFE problems, but this comparison deserves more attention given fixest's prominence in applied econometrics. The paper should:

- Include fixest in the performance benchmarks (Table 5)
- Clarify when users should prefer fixest over prompt2analytics for panel data applications

**Suggestion:** Add a fixest comparison row to Table 5 or explain in Section 5.6 why this comparison was not included.

---

## Minor Comments

### Code and Examples

1. **Page 22, CLI examples:** The backslash continuation syntax (`\`) may not work on all shells. Consider using the `--` convention or showing single-line commands.

2. **Page 19, Grunfeld example:** The paper states "This is a balanced panel dataset" but doesn't verify this programmatically. Consider showing how users can check panel balance.

3. **Section 4.2:** The CLI examples use `--session analysis.json` but never show how to resume a session or what happens if the session file already exists.

### Validation and Benchmarks

4. **Table 3:** The validation tolerances are appropriate, but the paper should note whether these were determined empirically or based on prior literature (e.g., NIST reference datasets).

5. **Section 5.4:** The explanation of why R is faster for PCA (0.3x speedup) is helpful. Similar explanations would benefit other methods where R outperforms Rust.

6. **Figure 3:** The y-axis label "Execution Time (ms, log scale)" is slightly redundant with the log scale already visible. Consider just "Execution Time (ms)".

### Writing and Presentation

7. **Page 8, Table 1:** The comparison table is useful but "Code review" under Correctness for Econometrics AI is vague. What does this mean in practice?

8. **Page 21:** The `\citep{Hausman:1978}` appears to not have rendered correctly ("Hausman test \citep{Hausman:1978}").

9. **Section 2.9:** The implementation caveats section is excellent but could note approximate computation time for these unimplemented methods in R, to help users understand the practical gap.

10. **Page 44, Table 9:** The "Server (large data)" scenario recommends 128 GB RAM but this seems excessive for the described use case. Is this a typo?

### Technical Points

11. **Appendix G.4:** The validation tolerance table (Table G.1) should specify whether the stated tolerances are for coefficient estimates, standard errors, or both.

12. **Page 29:** The claim that sandwich extracts leverage from QR decomposition while prompt2analytics uses direct matrix multiplication is interesting but would benefit from a citation or more detailed explanation.

13. **Section 3.4:** The choice of SurrealDB for persistence is mentioned but not justified. Why SurrealDB over SQLite or other embedded databases?

---

## Questions for the Authors

1. Has the software been tested with very wide datasets (e.g., p > n genomics applications)? The memory discussion focuses on row count but high dimensionality presents different challenges.

2. How does the system handle datasets with non-ASCII column names or special characters?

3. Is there any plan to support R's formula syntax as an alternative input method for users familiar with that notation?

---

## Typos and Minor Corrections

- Page 3: "ephemeral virtual machines" should perhaps be "sandboxed environments" for consistency
- Page 7: "matching and IPW" missing article before "matching"
- Page 21: Citation rendering issue mentioned above
- Page 42: "ref_level" appears in code font but isn't defined elsewhere
- Page 64: "$G \to \infty$" doesn't render as math in the warning message

---

## Overall Assessment

This is a solid software paper that makes a genuine contribution to the field. The chat-first paradigm is well-motivated, the implementation appears sound, and the validation is thorough. The main concerns relate to reproducibility of the LLM evaluation and some gaps in the comparison with existing tools. With minor revisions to address these points, the paper would be suitable for publication in JSS.

**Recommendation: Minor Revision**
