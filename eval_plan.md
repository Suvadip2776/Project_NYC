# NYCC OCT — Evaluation Plan

Evaluation of the NYCC character-trained model, comparing it against baselines to
answer one core question: **does character training buy us anything over simply
prompting the base model with the constitution?**

## Win condition

The key comparison is **fine-tuned + unprompted** vs. **base + prompted**.
If the trained model matches or beats the prompted base *without* the constitution
in its context, the character is genuinely internalized. That means it:
- frees up the context window (no need to spend tokens on the constitution),
- is more likely to hold under long-context and multi-turn interactions,
- is harder to strip out via prompt injection.

## EigenBench matrix (NYCC constitution)

Population = {student base, student fine-tune, teacher, 3 frontier reference models}
× {prompted, unprompted} = **6 × 2 = 12 runs**.

- **Unprompted** = baseline system prompt only (no constitution).
- **Prompted** = baseline system prompt **+ full NYCC constitution in context**.

Student-specific rows (the most informative subset):

| Model                     | Constitution in prompt? | What it tells us |
|---------------------------|-------------------------|------------------|
| Student base              | no (unprompted)         | floor / default behavior |
| Student base              | yes (prompted)          | **the prompting baseline** — does context alone suffice? |
| Student fine-tuned (OCT)  | no (unprompted)         | did training internalize the character without being told? |
| Student fine-tuned (OCT)  | yes (prompted)          | training + prompt compound (ceiling; check they don't conflict) |

Models:
- Student base / fine-tune: `Qwen/Qwen3.6-27B` (pre- and post-OCT). Fine-tune = `sdananya/qwen3.6-27b-nycc` (merged-final).
- Teacher: GLM-5.2 (`zai-org/GLM-5.2`)
- Reference (3 frontier): **gpt-4.1** (OpenRouter), **gemini-2.5-pro** (OpenRouter), **claude-sonnet-4.5** (direct Anthropic API)

## Additional evaluations

1. **Side-effect check** — EigenBench on a set of *reference* constitutions (e.g. the
   anchor constitutions from project 1 and/or the OCT constitutions) to confirm NYCC
   training didn't distort unrelated behavior.
2. **Capabilities evals** — confirm general capability did not regress relative to the
   pre-OCT baseline (lighteval / standard benchmarks).

## Fairness / methodology notes

- Apply the **identical** "prompted" format to every model so comparisons are fair
  (same constitution text, same placement, same baseline system prompt).
- **Teacher caveat:** the teacher already "knows" the constitution via its data-gen
  role-play prompt — do not give it an unfair advantage; use the same eval prompting
  as everyone else.

## Execution details (learned this session)

- **Scenarios: held-out, not the training prompts.** Use `kellycyy/AIRiskDilemmas` (what the
  OCT persona matrix used); optionally add AskReddit / OpenAssistant. Evaluating on the 500 NYCC
  training prompts is in-distribution leakage — avoid.
- **Serving:** base + fine-tune via vLLM (fine-tune needs the **vision-weight graft**,
  `scripts/graft_vision.py`, to be vLLM-servable; TP<=4 since kv_heads=4). Teacher GLM-5.2 via
  vLLM needs **8xH200** (TP=8/pp=1/fp8/non-eager). Frontier via API.
- **Prompted vs unprompted:** unprompted = no system prompt; prompted = full NYCC constitution as
  the system prompt (identical text/placement for every model).
- **Judges:** ideally full population (EigenBench design); pragmatic fallback = 3 frontier models
  as judges (reliable tag emission). Note the deviation.
- **Harness:** `scripts/gen_resp_ours.py`, `gen_resp_api.py`, `build_evaluations.py`
  (→ EigenBench `evaluations.jsonl`), then `EigenBench/scripts/run_train.py runs/nycc_full/spec.py`
  (BTD + EigenTrust + bootstrap). All in `nycc_repro/`.

### v1 (this session) — subset preview
6-model held-out run: base, base-prompted, fine-tune (unprompted) + 3 frontier (unprompted),
frontier-only judges, on AIRiskDilemmas. **TODO to reach full 12:** add teacher (GLM-5.2) +
prompted variants for fine-tune / teacher / 3 frontier; consider full-population judging.

## Decisions made
- **Baseline system prompt = none/empty.** Prompted = baseline + full NYCC constitution.
- **3 frontier models** = gpt-4.1, gemini-2.5-pro, claude-sonnet-4.5.

## Remaining open questions
- [ ] Judges = full population or frontier-subset for the final numbers?
- [ ] **Reference constitutions** for the side-effect check (project-1 anchors vs. OCT vs. both)?
- [ ] Capability benchmark suite + size (MMLU/GSM8K/IFEval/TruthfulQA subset?).
- [ ] Do eval scenarios need **images** (Qwen3.6-27B has a vision encoder) and/or
  **tool-use / MCP** and **long-context / multi-turn** coverage? *(from project notes)*
