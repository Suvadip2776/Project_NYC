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
- Student base / fine-tune: `Qwen/Qwen3.6-27B` (pre- and post-OCT)
- Teacher: GLM 5.2
- Reference: 3 frontier models (TBD)

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

## Open questions

- [ ] **Baseline system prompt** — what exactly goes in the "baseline" slot? Everything
  is measured relative to it, so it must be fixed before any runs. *(Question for Suvadip.)*
- [ ] Which **3 frontier reference models**?
- [ ] Which **reference constitutions** for the side-effect check (project-1 anchors vs. OCT)?
- [ ] Do eval scenarios need **images** (Qwen3.6-27B has a vision encoder) and/or
  **tool-use / MCP** and **long-context / multi-turn** coverage? *(from project notes)*
