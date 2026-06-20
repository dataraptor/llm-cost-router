# eval/data: benchmark slices + frozen split

Small offline data slices used by the no-key tests and the `--quick` eval path,
so neither needs network access.

## Files

- `gsm8k.jsonl`: 40 grade-school math word problems, one JSON object per line,
  `{"id", "question", "gold"}` where `gold` is the numeric answer.
- `mmlu.jsonl`: 40 four-option multiple-choice questions,
  `{"id", "subject", "question", "gold"}` where `gold` is the correct letter A-D.
- `frozen_split.json`: the deterministic ~20% test split, recorded for
  reproducibility. It is **derived** from `frugalroute.benchmarks.frozen_split`
  (a stable per-id hash, not an RNG shuffle); `core/tests/test_split.py` asserts
  the derivation still reproduces these exact ids, so the manifest cannot silently
  drift from the code.

## Provenance & license

These are **original items authored for FrugalRoute's demo and tests**, written in
the *style* of GSM8K (grade-school math) and MMLU (4-option multiple choice). They
are **not** drawn from the official GSM8K or MMLU datasets, so there is no upstream
license to carry and the repo stays self-contained and offline. The GSM8K golds are
computed in code at generation time (arithmetic correct by construction); the MMLU
golds are hand-verified factual answers.

They are intentionally small and representative, enough to exercise the loaders,
graders, frozen split, and a `--quick` eval, but not a statistically rigorous
benchmark. Swapping in larger, properly-licensed slices later only requires
replacing these JSONL files (same schema); the frozen split recomputes
deterministically.
