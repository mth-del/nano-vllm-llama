# Optimization Log

Track every optimization change in this file.

## How To Use

- Add one entry for each optimization.
- Keep entries in reverse chronological order (newest first).
- Include measurable impact when possible.

## Entry Template

```md
## [YYYY-MM-DD] <Title>
- Scope:
- Change:
- Why:
- Impact:
- Validation:
- Notes:
```

## Entries

## [2026-04-28] Llama3.2 RoPE scaling and dual examples
- Scope: RoPE compatibility and developer examples
- Change: Enhanced `rotary_embedding` with rope scaling parameters and added Llama3-style scaling support; updated `example.py` to include both Qwen and Llama runnable examples.
- Why: Improve Llama 3.2 long-context compatibility and make cross-model validation easier.
- Impact: More robust Llama RoPE behavior and simpler local smoke testing across model families.
- Validation: Pending runtime verification on real Llama3.2 checkpoint.
- Notes: Current implementation supports `rope_type=llama3` piecewise scaling path and keeps backward compatibility for existing models.

## [2026-04-28] Llama2/3.2 adaptation
- Scope: Model architecture compatibility (`Qwen3` -> `Llama2/3.2`)
- Change: Added `nanovllm/models/llama.py` and enabled runner-level model factory selection by `hf_config.model_type` (`qwen3` / `llama`).
- Why: Expand framework usability beyond Qwen3 and enable mainstream Llama inference workloads.
- Impact: Framework can now instantiate a dedicated Llama model path and load Llama-style packed weights.
- Validation: Passed `python3 -m compileall nanovllm` syntax validation.
- Notes: Llama 3.2 long-context `rope_scaling` behavior may still need follow-up tuning for strict parity.

## [2026-04-28] Initialize optimization log
- Scope: Project documentation
- Change: Added `OPTIMIZATION_LOG.md` with a standard entry template.
- Why: Keep optimization history traceable and easy to review.
- Impact: Establishes a single source of truth for future performance updates.
- Validation: N/A (documentation-only change).
- Notes: Add one new entry whenever an optimization is implemented.
