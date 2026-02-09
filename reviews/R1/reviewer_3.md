# Review: Chat-First Data Analytics

**Reviewer 3**
**Expertise:** Machine Learning, LLM Applications, Software Interfaces
**Recommendation:** Minor Revision

---

## Summary

This paper presents prompt2analytics, a system that combines a Rust econometrics library with LLM-based natural language interfaces. From an ML/NLP perspective, the paper makes a compelling case for tool-augmented LLM architectures in specialized domains. The evaluation is thorough and refreshingly honest about limitations. I recommend minor revision.

---

## Major Comments

### 1. Extended Evaluation is a Strength, But Presentation Could Improve

The extended robustness evaluation (Section 7.3, Appendix B) is one of the paper's strongest contributions. Unlike many LLM papers that report only best-case accuracy, this work honestly examines multi-turn degradation, parameter extraction failures, and out-of-scope detection. However:

- Table 6 is buried on page 40; key findings should appear earlier
- Figure B.1 (radar chart) effectively visualizes the multi-dimensional performance but appears only in the appendix
- The "gap between controlled and realistic conditions" narrative could be highlighted more prominently

**Suggestion:** Create a condensed version of Figure B.1 for the main text, and move Table 6 content to Section 7 proper.

### 2. Model Selection for Local Deployment Needs More Justification

Table 7 recommends specific models for local deployment (Qwen 2.5 72B, Llama 3.3 70B, Ministral 3B), but the evaluation basis is the 87-test single-turn benchmark. Given that:

- Multi-turn accuracy differs substantially across models (Table B.1)
- Parameter extraction F1 varies from 41-59%
- Different models excel on different dimensions

The recommendations may not generalize to actual usage patterns. A user doing exploratory analysis (multi-turn) might prefer a different model than one doing batch single-turn analyses.

**Suggestion:** Provide differentiated recommendations based on expected use case, or caveat the current recommendations more strongly.

### 3. Prompt Engineering and System Prompts

The paper doesn't disclose the system prompts used for LLM evaluation. For reproducibility and to help practitioners, the paper should:

- Include the system prompt(s) in supplementary materials
- Discuss whether prompt engineering was attempted and what worked/didn't
- Note whether different prompts were needed for different model families

**Suggestion:** Add system prompt details to Appendix A or create a new supplementary file.

---

## Minor Comments

### LLM Evaluation

1. **Temperature settings:** The evaluation methodology should specify temperature and other generation parameters. This affects reproducibility.

2. **Model versioning:** Table A.1 lists models but not exact version identifiers. Given rapid model iteration (GPT-4o has had multiple versions), this matters for reproducibility.

3. **Context length:** How does the system perform when context approaches the model's limit? This is relevant for multi-turn conversations with complex analyses.

4. **Figure 8:** The accuracy comparison chart is excellent. Consider adding confidence intervals or error bars if multiple evaluation runs were conducted.

5. **Out-of-scope detection (40-60%):** This is notably weak. The paper should discuss implications: users may get plausible-looking but inappropriate results for unsupported analyses.

### Tool Schema Design

6. **Tool count:** 101 tools is substantial. Has any analysis been done on whether tool selection accuracy degrades with schema size? The Gorilla paper (Patil et al. 2023) suggests this can be an issue.

7. **Tool descriptions:** The paper mentions tools are "written to be interpretable by LLMs" (p.16) but doesn't elaborate. What makes a tool description LLM-friendly? This would be valuable guidance for practitioners.

8. **Schema evolution:** How will the API evolve? If tool schemas change, will this break LLM integrations? Some discussion of versioning strategy would be helpful.

### Local LLM Deployment

9. **Quantization effects:** Table 7 mentions Q4_K_M quantization, but does this affect tool selection accuracy? Some quantized models show degraded instruction-following.

10. **Ollama vs. other runtimes:** The paper focuses on Ollama, but other local inference solutions exist (llama.cpp directly, vLLM, etc.). A brief mention of alternatives would help readers.

11. **Latency considerations:** Figure C.1 shows cloud API latency. Local inference latency depends heavily on hardware and model size; this should be discussed.

### Writing and Presentation

12. **"Chat-first" terminology:** This is a nice framing, but I wonder if "NL-first" (natural language first) might be more precise. "Chat" implies conversation, but single-turn interactions are also covered.

13. **Section 2.3 limitations:** The acknowledgment that chat-first doesn't replace statistical expertise is important and well-stated. Consider making this even more prominent.

14. **Page 37, Figure 7:** The architecture diagram is clear. Consider adding a legend explaining the color coding.

15. **Appendix B.3 failure analysis:** The categorization of failure patterns (context drift, partial extraction, overconfident selection, interpretation shortcuts) is valuable. This could be promoted to the main text.

---

## Technical Suggestions

### For Improved Robustness

1. **Structured output:** Has the system been tested with structured output modes (e.g., JSON mode in OpenAI API)? This could improve parameter extraction reliability.

2. **Retry logic:** For the 9.2% failure rate with Ministral 3B, does the system support automatic retry with different phrasing?

3. **Confidence scores:** Could the system expose LLM confidence (via logprobs) to warn users when tool selection is uncertain?

### For Extended Evaluation

4. **Adversarial prompts:** Has robustness to adversarial or confusing prompts been evaluated? Users may accidentally phrase requests ambiguously.

5. **Cross-lingual:** Does the system work with non-English prompts? Given international usage of econometrics, this could be valuable.

---

## Questions for Authors

1. How were the 87 test cases developed? Were they written by domain experts, derived from documentation, or collected from users?

2. The multi-turn evaluation uses 20 conversations with 72 total turns. How were these conversations designed? Do they represent realistic analysis workflows?

3. Has any A/B testing or user preference data been collected comparing chat-first to traditional interfaces?

4. Given the parameter extraction challenges, have you considered hybrid approaches where the LLM proposes parameters and users confirm/modify?

---

## Strengths Worth Highlighting

- **Honest evaluation:** The acknowledgment that single-turn accuracy overstates practical capability is commendable
- **Local deployment focus:** The emphasis on data privacy through local LLMs addresses a real concern
- **Comprehensive method coverage:** 200+ methods is genuinely impressive for a new library
- **Thorough validation:** The comparison against R reference implementations is rigorous

---

## Overall Assessment

This paper makes a solid contribution at the intersection of LLM applications and statistical computing. The main strengths are the comprehensive Rust library, honest evaluation including failure modes, and practical focus on local deployment. The main weakness is the presentation, which buries important limitations in appendices while the introduction and examples suggest a more polished experience than the evaluation supports.

With minor revisions to improve the presentation of limitations and add details about prompts/evaluation methodology, this would be a valuable JSS publication.

**Recommendation: Minor Revision**
