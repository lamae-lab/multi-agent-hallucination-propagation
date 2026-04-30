# Project Plan: Hallucination Resilience Across Multi-Agent Architectural Families

**Target venue:** AGENTICS 2026 (IJCCI 2026), Angers, France, 28–30 October 2026
**Springer CCIS proceedings, indexed in Scopus, DBLP, Web of Science**

**Submission targets:**
- Primary: 19 May 2026 (regular paper main track)
- Fallback: 26 June 2026 (position/regular paper)

**Author:** Samir Jabbar (samir.jabbar17@gmail.com)
**Last updated:** 2026-04-29

---

## 1. Context and Motivation

The agent ecosystem has converged in 2025–2026 around four major frameworks (LangGraph, CrewAI, AutoGen/AG2, OpenAI Swarm / Agents SDK) and one canonical single-agent pattern (ReAct with tools). Each implements a fundamentally different architectural family for organizing multi-LLM coordination.

**Practitioner question:** *Which architecture should I choose to minimize hallucination risk?*

No systematic empirical study compares hallucination behavior across these architectural families. Existing work either:
- Studies hallucinations within a single agent paradigm (post-hoc attribution, single-trajectory analysis), or
- Compares frameworks on speed, cost, or developer ergonomics — never on hallucination resilience.

This project provides the first systematic comparison of how hallucinations propagate through each architectural family, under controlled experimental conditions, with replication on safety-critical medical reasoning.

---

## 2. Research Questions

**Main RQ:** How do the major multi-agent architectural families compare in terms of hallucination resilience and error propagation?

**Sub-questions:**
- **RQ1:** Which architectural family produces the lowest final-output hallucination rate on controlled tasks?
- **RQ2:** When a hallucination is introduced at one point in each architecture, what is its survival rate? How does each family contain or amplify it?
- **RQ3:** Does each family exhibit a characteristic failure mode (self-validation, state freeze, manager bias, consensus, chain of trust)?
- **RQ4:** Is there a Pareto frontier between hallucination resilience and operational cost (tokens, latency)?
- **RQ5:** Do these patterns transfer to safety-critical domains? (Medical reasoning study, Section 9.)

---

## 3. Hypotheses

- **H1:** Architectural families differ significantly in hallucination resilience, with effect size large enough to inform practitioner decisions.
- **H2:** F1 (Single-Agent ReAct) shows higher Self-Validation Rate than other families exhibit for their respective failure modes.
- **H3:** F4 (Group-Chat) exhibits Consensus Rate growth with conversation length.
- **H4:** F2 (State-Graph) shows highest State-Freeze Rate.
- **H5:** F5 (Handoff Swarm) shows high Trust-Chain Rate and low Detection Rate.
- **H6:** No single family dominates; Pareto frontier exists across resilience-vs-cost.
- **H7:** Family rankings on general-purpose tasks transfer to MedQA, but failure severity is amplified.

---

## 4. Architectural Families

Each family is a faithful reproduction of its dominant framework's pattern. **Implementation choice (real frameworks vs minimal reimplementation) is decided by the Phase 1 spike test** — we use real frameworks where Ollama compatibility is verified, reimplement only where it fails, with documented fidelity validation in either case.

### F1 — Single-Agent ReAct
- **Reference:** OpenAI Assistants SDK, Claude SDK with tools, Hermes-Agent
- **Communication pattern:** none (single agent in Think → Act → Observe loop)
- **Hypothesized failure mode:** self-validation loops
- **Role in study:** practitioner baseline (not architectural competitor — see C2 in `/.claude/plans/for-the-timeline-we-majestic-brooks.md`). Reported separately from cross-family pairwise tests.

### F2 — Sequential State-Graph
- **Reference:** LangGraph
- **Communication pattern:** indirect, mediated by structured shared state
- **Hypothesized failure mode:** state freeze (hallucination written to state read as ground truth by all subsequent nodes)

### F3 — Hierarchical Crew
- **Reference:** CrewAI
- **Communication pattern:** hub-and-spoke; manager dispatches to specialists
- **Hypothesized failure mode:** manager bias (orchestrator amplifies/filters idiosyncratically)

### F4 — Conversational Group-Chat
- **Reference:** AutoGen / AG2 GroupChat
- **Communication pattern:** shared conversational history
- **Hypothesized failure mode:** consensus-building around hallucination (conformity)

### F5 — Autonomous Handoff Swarm
- **Reference:** OpenAI Swarm / OpenAI Agents SDK
- **Communication pattern:** handoff with summarized context, no central orchestrator
- **Hypothesized failure mode:** chain of trust (receiver over-trusts handoff context)

### Common interface
```
solve(task: Task) -> Result
```
With structured trace `List[Message]` of all inter-agent communication. Same underlying LLM across families. YAML config per family. Deterministic seeding. Each family small enough to inspect (200–400 LOC if reimplemented).

---

## 5. Tasks and Datasets

### 5.1 Main Study — General-Purpose
- **HotpotQA** (200–500 questions, distractor setting): multi-hop QA, naturally decomposable.
- **GSM8K** (200–300 problems): mathematical reasoning, unambiguous ground truth.

### 5.2 Hallucination Source — HaluEval
Used as a bank of validated hallucinations for naturalistic injection. QA, dialogue, summarization splits (~200–300 examples).

### 5.3 Applied Study — Medical Reasoning
**MedQA** (USMLE-style MCQ, 200–300 questions). Replicates main experiment with domain-appropriate hallucinations (wrong drug dosage, false contraindication, fabricated guideline citation).

### 5.4 Dataset Rollout
1. All infrastructure on HotpotQA (canonical test bed).
2. Add GSM8K once Phase 3 metrics are validated.
3. Integrate HaluEval injection in Phase 4.
4. Add MedQA last, only after main study is complete.

---

## 6. Hallucination Taxonomy and Injection Protocol

### 6.1 Hallucination Taxonomy
Adopted from arxiv 2509.18970 (Sept 2025), "LLM-based Agents Suffer from Hallucinations: A Survey":

1. **Reasoning hallucinations** — flawed plans (goal misunderstanding, bad decomposition)
2. **Execution hallucinations** — claiming a sub-task was performed when it wasn't (tool selection, tool calling)
3. **Perception hallucinations** — observations misrepresenting the environment
4. **Memorization hallucinations** — trusting memory without verification (retrieval, update)
5. **Communication hallucinations** — fabricated/misleading inter-agent info

Categories 4 and 5 are most relevant: Communication maps to inter-agent propagation; Memorization maps to F2's state-freeze hypothesis.

Mapping from previous types:
- Factual → Memorization or Communication
- Attribution (fake source) → Communication
- Completeness (claimed-not-done) → Execution
- Subtle → cross-cuts all categories

### 6.2 Injection Modes

**Mode 1 — Synthetic Controlled (primary).** Force a specific agent to produce a specific false statement via prompt engineering. *Pro:* total causal control. *Con:* experimenter-designed.

**Mode 2 — HaluEval-Sourced (validation).** Use HaluEval's validated hallucinated answers as the forced output of the first agent. *Pro:* ecologically valid. *Con:* less control over type/severity.

**Mode 3 — Naturalistic Adversarial (robustness check, optional).** Use TruthfulQA-style queries; observe spontaneous propagation. *Pro:* most realistic. *Con:* requires post-hoc verification. **Decision:** include only if budget allows; otherwise drop.

### 6.3 Injection Strategy
- Mode 1 = primary, cleanest causal claims.
- Mode 2 = validates generalization to realistic hallucinations.
- For multi-agent: inject at the first agent in natural processing order. For F1: inject at first iteration.

---

## 7. Metrics

Restructured into two layers (see C3 in plan file).

### 7.1 Layer 1 — Established Metrics (for reviewer credibility)
- **Faithfulness** (RAGAS-style atomic-claim verification): does final output contain claims unsupported by ground truth?
- **Answer correctness / task success:** does the final output answer correctly?
- **Hallucination rate** with 95% CI per family.
- **FActScore-style** atomic-claim decomposition for free-form outputs.

Citations to ground these: RAGAS, FActScore, HalluLens (ACL 2025), TruthfulQA, SelfCheckGPT.

### 7.2 Layer 2 — Novel Contributions
- **Survival Rate:** is the injected hallucination still in the final output? Detected via exact match + semantic similarity (embedding cosine) + LLM-as-judge. Graded: verbatim / paraphrased-but-present / contradicted / absent. Report inter-method agreement.
- **Propagation Distance:** number of agents/iterations the hallucination survives before being corrected/contradicted/dropped.
- **Amplification Factor:** does confidence in hallucinated content increase as it propagates? Measured via:
  - Linguistic confidence markers (hedging vs categorical assertions, LLM-judge)
  - Removal of uncertainty qualifiers between consecutive agents
  - **Drop self-reported confidence** — LLMs are not calibrated.
- **Mutation Rate:** how many agents reformulate the hallucinated content (laundering)? Measured via semantic similarity between original injection and each agent's restatement.
- **Detection Rate:** for configurations with verifier agents, fraction of injections caught.

### 7.3 Family-Specific Behavioral Markers (normalized — see C4)
- **F1 Self-Validation Rate** = (iterations reasserting hallucination) / (total iterations)
- **F2 State-Freeze Rate** = (nodes reading hallucinated state without question) / (total downstream nodes)
- **F3 Manager-Amplification** = (severity of manager output) − (max severity across specialists), normalized to [-1, 1]
- **F4 Consensus Rate** = (agents agreeing with hallucination) / (total agents)
- **F5 Trust-Chain Rate** = (un-verified handoffs) / (total handoffs)

**Explicit caveat in paper:** these markers are diagnostics for each family's hypothesized failure mode and are **not directly comparable across families**. Cross-family comparisons use Layer 1 + family-neutral Layer 2 metrics.

### 7.4 Reporting (replaces ad-hoc Containment formula)
- 2D scatter: Survival Rate × Amplification Factor per family.
- Pareto plot: hallucination resilience × cost (tokens, latency, $).
- Heatmap: Family × Hallucination Type interaction.
- Propagation diagrams per family (visual case studies).

### 7.5 Cost Metrics
- Tokens per task (input + output, all agents combined)
- Wall-clock latency
- $ cost per task

### 7.6 Inter-Rater Reliability
For LLM-as-judge: run **two judge models** (e.g., GPT-4o + Claude Haiku, or two different Ollama models) on the same outputs, report **Cohen's κ**. Reviewers will ask.

---

## 8. Experimental Design

### 8.1 Independent Variables
- Architectural family (5 levels): F1, F2, F3, F4, F5
- Hallucination type (5 categories from 6.1; primary + 4 sub-types)
- Injection mode (2–3 levels): synthetic / HaluEval-sourced / [naturalistic optional]
- Underlying LLM (1–2 levels via Ollama; expand if budget allows)
- Task: HotpotQA / GSM8K / MedQA (applied study)

### 8.2 Sample Size
- Minimum 30–50 task instances per condition
- 3 seeds per instance (more if budget allows)
- **Power analysis deferred to before Phase 4 lock** (see C5).

### 8.3 Controls
- Same input task across families (paired comparison).
- Same underlying LLM across compared families.
- Same tools available to all agents.
- Random seed control.
- Temperature fixed at 0 for main study.
- Frozen agent prompts per family, documented and reviewed.

### 8.4 Statistical Analysis
- Paired comparisons (Wilcoxon signed-rank or paired t-test).
- Mixed-effects models for nested designs.
- Multiple-testing corrections (Holm or Benjamini-Hochberg).
- Effect sizes (not just p-values).
- F1 reported separately as baseline (not in pairwise cross-family tests).

---

## 9. Applied Study — Medical Reasoning

After completing main study, replicate on MedQA.

### 9.1 Goals
- Test whether family ranking persists in medical reasoning.
- Identify domain-specific aggravations.
- Provide impact argument linking architectural choices to safety-critical risks.

### 9.2 Methodology
- Same five architectures, same metrics, same injection protocol.
- Domain-appropriate hallucinations (wrong drug dosage, false contraindication, fabricated guideline citation).
- 200 MedQA questions × 3 seeds × 5 families.

### 9.3 Expected Outcomes
- Ranking holds → strengthens generalizability claim.
- Ranking shifts → finding of its own (domain-specific recommendations).

### 9.4 Cuttable
This section is structurally separable. Can be cut without harming main contribution if timeline pressure forces it.

---

## 10. Project Structure

```
halucinations_agents/
├── README.md
├── pyproject.toml
├── docs/
│   ├── PROJECT_PLAN.md          # this file
│   ├── BRIEF_v3.md              # archived original brief
│   └── CHANGELOG.md             # decision log
├── configs/
│   ├── families/                # YAML per architectural family
│   ├── injections/              # injection prompts and target hallucinations
│   └── tasks/                   # task configurations
├── src/
│   ├── agents/                  # base agent class, specialist agents
│   ├── families/                # f1_react.py, f2_state_graph.py, ...
│   ├── injection/               # injection protocols (modes 1, 2, 3)
│   ├── metrics/
│   │   ├── layer1_established.py    # Faithfulness, FActScore, task success
│   │   ├── layer2_propagation.py    # Survival, Propagation, Amplification, Mutation
│   │   └── family_markers.py        # F1–F5 normalized markers
│   ├── tasks/                   # HotpotQA, GSM8K, HaluEval, MedQA loaders
│   ├── llm/                     # LLM client (Ollama-first, OpenAI-compat)
│   ├── tracing/                 # JSONL trace serialization
│   ├── analysis/                # statistical analysis, plotting
│   └── ui/                      # experiment dashboard (Phase 4.5)
├── experiments/
│   ├── phase1_spike/            # Ollama-compat spike test
│   ├── run_main_study.py
│   ├── run_medical_study.py
│   ├── configs/
│   └── results/                 # raw run outputs (JSONL)
├── notebooks/                   # exploratory analysis, figure generation
└── tests/
```

---

## 11. Implementation Phases

### **Phase 0 — Project setup** *(1 day)*

**Deliverables:**
- `pyproject.toml` with deps: `ollama`, `litellm`, optional `langgraph`, `crewai`, `autogen`, `openai-agents`.
- Repo skeleton matching Section 10 layout.
- `docs/CHANGELOG.md` to log methodology decisions.
- `.gitignore`, ruff/black config.

**Done when:** clean Python env runs, deps install, repo structure matches plan.

---

### **Phase 1 — Ollama framework spike test** *(1–2 days, blocks Phase 2)*

**Goal:** Decide real-frameworks vs reimplementation (resolves C1).

**Deliverables:**
- `experiments/phase1_spike/spike_langgraph.py` — 2-agent + 1-tool hierarchy on 5 HotpotQA questions.
- `experiments/phase1_spike/spike_crewai.py` — same task, CrewAI.
- `experiments/phase1_spike/spike_autogen.py` — same task, AutoGen/AG2 GroupChat.
- `experiments/phase1_spike/spike_openai_agents.py` — same task, OpenAI Agents SDK with handoffs.
- `experiments/phase1_spike/RESULTS.md` — per-framework: did it run? tool-call success rate? trace inspection sensible?

**Decision rule (recorded in `docs/CHANGELOG.md`):**
- ≥3 of 4 frameworks work cleanly → use real frameworks for those, document fidelity for any reimplemented.
- ≤2 work → mixed approach with per-framework fidelity validation studies.
- 0 work with chosen Ollama model → escalate to larger model first; reimplementation is last resort.

**Done when:** decision recorded with evidence in CHANGELOG.

---

### **Phase 2 — Core infrastructure** *(3–5 days)*

**Goal:** Foundation for all subsequent phases.

**Deliverables:**
- `src/llm/client.py` — Ollama-first LLM client (OpenAI-compatible endpoint), with model abstraction so we can swap models later.
- `src/agents/base.py` — Base Agent class with tool-use support; uniform interface across families.
- `src/tracing/trace.py` — JSONL serialization: full inputs, all agent messages, outputs, configs, seed, model version. Reproducibility-first.
- `src/tasks/hotpot_qa.py` — HotpotQA loader with deterministic sampling.
- `tests/` — basic unit tests on tracing and LLM client.
- Simple HTML viewer for trace inspection (optional but valuable for case review).

**Done when:** can run a single agent on 1 HotpotQA question end-to-end, get a complete JSONL trace.

---

### **Phase 3 — Five architectural families** *(7–10 days)*

**Goal:** All 5 families implemented per Phase 1 decision.

**Deliverables:**
- `src/families/f1_react.py`
- `src/families/f2_state_graph.py`
- `src/families/f3_crew.py`
- `src/families/f4_group_chat.py`
- `src/families/f5_handoff_swarm.py`
- For each: YAML config in `configs/families/`, validation on 5–10 HotpotQA tasks, structured trace verifying inter-agent communication is captured.
- Paired comparison harness: feed same input through all five, save aligned traces.

**Done when:** all 5 families produce valid traces on the same HotpotQA task; manual inspection confirms architectural pattern is recognizable.

---

### **Phase 4 — Injection Mode 1 + Layer 2 metrics + pilot** *(5–7 days)*

**Goal:** Working pipeline with synthetic injection and core metrics.

**Deliverables:**
- `src/injection/mode1_synthetic.py` — prompt-engineering injection at first agent.
- `src/metrics/layer2_propagation.py` — Survival, Propagation Distance, Amplification, Mutation Rate.
- `src/metrics/family_markers.py` — normalized F1–F5 markers per Section 7.3.
- Multi-method detection: exact match + semantic similarity + LLM-as-judge. Report inter-method agreement.
- **Pilot run:** 10–20 tasks × 5 families × Mode 1 → first numerical results, sanity check.
- `notebooks/phase4_pilot.ipynb` — exploratory analysis of pilot.

**Done when:** pilot produces sensible Survival × Amplification scatter; no crashes; metrics are reproducible across reruns.

---

### **Phase 4.5 — Experiment Dashboard UI** *(5–7 days)*

**Goal:** Web-based dashboard for running experiments, viewing traces, and analyzing metrics interactively. Built after Phase 4 so all metrics and injection are available from day one.

**Deliverables:**
- `src/ui/` — Web application (Streamlit or Gradio) for the experiment dashboard.
- **Experiment Runner**: Select families (any combination via checkboxes), pick dataset and number of questions, configure injection mode and parameters, run experiments with progress tracking.
- **Trace Viewer**: Side-by-side trace comparison across families for the same question, message flow visualization with tool calls highlighted, expand/collapse individual agent messages.
- **Metrics Dashboard**: Survival × Amplification scatter plots, propagation distance per family bar charts, family-specific behavioral markers (self-validation, state-freeze, manager-bias, consensus, trust-chain), heatmap of Family × Hallucination Type interaction.
- **Family Comparison**: Select any 2–5 families, compare answers, metrics, and traces on the same questions. Diff view showing where answers diverge.
- **Reports**: Per-family summary cards, export results to CSV/LaTeX for paper, aggregate statistics across experiment runs.
- **Data Management**: Browse saved experiment runs, upload custom questions, manage HotpotQA/GSM8K/HaluEval samples.

**Done when:** Dashboard loads, can run a 5-family × 5-question experiment from the UI, displays metrics and traces, exports results.

---

### **Phase 5 — Layer 1 metrics + main study scale-up** *(7–10 days)*

**Goal:** Full main study on HotpotQA + GSM8K with all metric layers.

**Deliverables:**
- `src/metrics/layer1_established.py` — Faithfulness (atomic-claim), task success, FActScore-style decomposition.
- LLM-as-judge with **two judge models** for inter-rater reliability (Cohen's κ reported).
- `src/tasks/gsm8k.py` — GSM8K loader.
- `src/injection/mode2_halueval.py` — HaluEval-sourced injection.
- `src/tasks/halueval.py` — HaluEval loader.
- **Power calculation** (resolves deferred C5) → final n per condition locked.
- Full main study run: HotpotQA + GSM8K × 5 families × Modes 1+2 × 30–50 tasks × 3 seeds.
- Statistical analysis with Holm/BH correction, effect sizes, paired tests.

**Done when:** all main-study results in `experiments/results/` as JSONL; statistical analysis script runs end-to-end and produces tables/figures.

---

### **Phase 6 — Applied study (MedQA)** *(5–7 days)*

**Goal:** Replicate main study on medical reasoning.

**Deliverables:**
- `src/tasks/medqa.py` — MedQA loader.
- `configs/injections/medical.yaml` — domain-appropriate hallucinations (drug dosage, contraindication, fabricated guideline).
- `experiments/run_medical_study.py` — replication script.
- Comparison analysis: does family ranking from main study persist on MedQA?

**Done when:** MedQA results integrated; cross-study comparison figure generated.

---

### **Phase 7 — Analysis, figures, writing** *(7–10 days)*

**Goal:** Camera-ready paper.

**Deliverables:**
- All figures: Pareto plots, family-comparison heatmaps, propagation diagrams, Survival × Amplification scatters.
- `notebooks/figures.ipynb` — figure generation.
- Paper draft (LaTeX) following AGENTICS 2026 / Springer CCIS template.
- Supplementary: full trace dataset, configs, instructions to reproduce.
- Internal review pass.

**Done when:** PDF compiles, figures embedded, supplementary uploaded, ready for submission.

---

## 12. Engineering Requirements

- **Reproducibility:** every run saves complete trace (inputs, all agent messages, outputs, configs, seed, model version) as JSONL.
- **Cost control:** cost estimator before expensive runs; response caching where deterministic.
- **Modularity:** swapping LLM, family, or injection mode requires no rewrites.
- **Observability:** per-run trace inspectable manually + simple HTML viewer.
- **Statistical rigor:** paired comparisons, multiple-testing corrections, reported effect sizes.
- **Family fidelity:** each family implementation reviewable by someone familiar with the reference framework. If reimplemented, documented mapping + fidelity validation study.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Family implementations not faithful enough → strawman criticism | Phase 1 spike decides real-vs-reimpl; reimpls get documented mapping + fidelity validation study |
| Ollama tool-calling unreliable | Use vetted models (qwen2.5:14b+, llama3.1:70b); upgrade model before reimplementing |
| Injection feels artificial | Mode 2 (HaluEval) addresses this with realistic hallucinations |
| Results LLM-dependent | Replicate on ≥2 LLMs in sensitivity analysis (budget permitting) |
| Hallucination detection is fuzzy | Multi-method detection, report agreement, conservative intersection as headline |
| Too many factors → underpowered | Prioritize Family × Hallucination Type interaction; power analysis before Phase 5 lock |
| F1 baseline seems unfair | Reframed as practitioner baseline, separated from cross-family pairwise tests |
| MedQA bloats paper / time pressure | Section 9 structurally separable; can be cut |

---

## 14. Deliverables (final)

1. Reproducible codebase (deterministic seeds, configs, instructions).
2. Dataset of injection traces (publishable as supplementary).
3. Statistical analysis with main figures.
4. Camera-ready paper for AGENTICS 2026.

---

## 15. Positioning Statement

The contribution is methodological and empirical:

1. **First systematic comparison** of the five major multi-agent architectural families on hallucination resilience.
2. **Family-specific failure mode markers** linking observed behavior to architectural mechanisms (self-validation, state freeze, manager bias, consensus, chain of trust).
3. **Controlled experimental protocol** combining synthetic and HaluEval-sourced injection, enabling causal claims.
4. **Cross-domain transfer evidence** via MedQA replication.
5. **Actionable design guidance** via Pareto frontier analysis.

Complements existing hallucination-attribution research ("where did the hallucination come from?") and existing framework benchmarks ("how fast/cheap?") by addressing the orthogonal question: **"how does the choice of architectural family affect hallucination resilience?"**

---

## References (key citations)

- **Agent hallucination taxonomy:** "LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions" (arxiv 2509.18970, Sept 2025).
- **Hallucination benchmarks:** HalluLens (ACL 2025), TruthfulQA, HaluEval.
- **Established metrics:** RAGAS (Faithfulness, Answer Relevance), FActScore, SelfCheckGPT, G-Eval.
- **Frameworks:** LangGraph, CrewAI, AutoGen/AG2, OpenAI Agents SDK.
- **Datasets:** HotpotQA, GSM8K, MedQA.

---

## Appendix: Decision Log Pointer

All methodological decisions (real-frameworks vs reimplementation, final n per condition, dropped Mode 3 if applicable, etc.) recorded in `docs/CHANGELOG.md` as they happen, with date and rationale.
