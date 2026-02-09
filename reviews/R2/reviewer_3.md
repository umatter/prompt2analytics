# Review: Chat-First Data Analytics (Revision)

**Reviewer 3**
**Expertise:** Machine Learning, LLM Applications
**Previous Recommendation:** Minor Revision
**Current Recommendation:** Accept

---

## Summary

The authors have comprehensively addressed my previous concerns. The revision substantially improves the paper's presentation of limitations, adds the reproducibility details I requested, and provides differentiated model recommendations. I recommend acceptance.

---

## Response to Previous Comments

### Major Comments

**3.1 Extended evaluation presentation**

The authors have promoted key findings to earlier sections as requested:
- New Section 2.4 presents Table 1 (accuracy summary) before the examples
- The abstract now includes accuracy metrics (47-60% multi-turn, 41-59% F1)
- Section 1.4 includes forward references noting accuracy drops

This addresses my concern that limitations were buried in appendices. Readers now encounter the realistic accuracy expectations before seeing the polished examples. **Satisfactorily addressed.**

**3.2 Model selection for different use cases**

The new subsubsection "Recommendations by use case" in Section 7.4 provides exactly the differentiation I requested:
- Single-turn analyses: smaller models (Ministral 3B) adequate
- Multi-turn workflows: larger models (Llama 3.3 70B) preferred for context maintenance
- Parameter-sensitive analyses: Qwen 2.5 72B for highest F1
- Interpretation: GPT-4.1 Nano for best explanation quality

This is practical guidance that users can act on. **Satisfactorily addressed.**

**3.3 Prompt engineering and system prompts**

The new "Evaluation methodology and reproducibility" subsubsection in Section 7.2 addresses my concerns:
- Temperature = 0 for all evaluations
- System prompt summary in text, full prompt in supplementary materials
- Exact model version identifiers with dates
- Acknowledgment of API model instability

The note that "different prompts were not needed for different model families" is useful information. **Satisfactorily addressed.**

### Minor Comments

**Temperature and model versioning:** Addressed in Section 7.2.

**"Chat-first" terminology:** The authors retained "chat-first" rather than switching to "NL-first." While I suggested this change, their reasoning (emphasizing conversational aspect that distinguishes from one-shot systems) is reasonable. The terminology is not a major issue.

---

## Assessment of New Content

### Section 2.4: Practical accuracy expectations

This addition transforms the paper's honesty about limitations. The table presenting:
- Single-turn: 90.8-100% (curated best case)
- Multi-turn: 47-60% (realistic use)
- Conversation completion: 0-35%
- Parameter extraction F1: 41-59%
- Out-of-scope detection: 40-60%

...followed by practical implications (expect corrections, verify parameters, use explicit specifications) is exactly what practitioners need. This section should be read by anyone considering deploying chat-first analytics systems.

### Section 4.6: Realistic multi-turn workflow

The three-turn failure example (t-test → DiD correction → clustered SE addition) effectively demonstrates what users should expect. The acknowledgment that "users should plan for such corrections" is honest and practical.

### Table 3: Method coverage

The coverage gap table with recommended alternatives is valuable. Users can quickly see that Bayesian inference needs rstanarm/brms, Heckman selection needs sampleSelection, etc.

### Evaluation methodology details

The reproducibility section provides sufficient detail for follow-up research. The exact model identifiers and temperature settings enable meaningful replication attempts, with appropriate caveats about API model instability.

---

## Technical Quality

The technical content remains strong:
- Comprehensive LLM evaluation across 10 models
- Extended robustness evaluation beyond single-turn accuracy
- Honest reporting of failure modes and limitations
- Clear documentation of what works and what doesn't

The evaluation methodology (87 test cases, 20 multi-turn conversations, naturalistic prompt variations) is thorough and well-documented.

---

## Remaining Minor Issues

None that would prevent acceptance. The paper is ready for publication.

---

## Overall Assessment

The revision addresses all my previous concerns:
1. Limitations are now prominently presented early in the paper
2. Model recommendations are differentiated by use case
3. Reproducibility details (prompts, temperature, model versions) are documented
4. The gap between controlled and realistic evaluation is clearly acknowledged

The paper makes an important contribution to understanding how LLMs can be integrated with validated analytical software. The honest presentation of accuracy limitations (particularly the 47-60% multi-turn accuracy) provides valuable guidance for practitioners considering similar systems.

**Recommendation: Accept**
