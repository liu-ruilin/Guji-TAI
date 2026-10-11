# Paper stability evaluation

`stability.span_iou` remains the mean Top-k evidence Jaccard overlap between
the original prompt and its evaluated variants. A genuine measured perfect
overlap remains `1.0`. When no distinct, nonempty variant is evaluated, the
value is `None` in Python / `null` in JSON, rather than a perfect score.
`diagnostics.stability_comparisons` records the number of comparisons per sample.
`variant_limit` still includes the original prompt; setting it to 1 disables
stability comparisons. Duplicate, blank, and original prompts are excluded
before applying the limit. Explicit variants may omit the original prompt.

Automatic variants preserve the original instruction, question and input or
context content, changing only separators or framing labels. This covers
translation, instruction-only QA, paraphrase, relation extraction, reasoning,
missing-character completion, and punctuation tasks without changing source
characters, masks, relation schemes, or the explanation target. Context-dependent
questions no longer get context-free variants. Opaque/custom prompts and bare
source text are not reconstructed from partial fields; supply explicit
target-consistent `prompts` to evaluate their stability. Explicit variants remain
the caller's responsibility. These lightweight perturbations measure framing
sensitivity; they do not establish robustness to broader semantic perturbations.

Dataset summaries average each metric over its finite observed values, excluding
`null`, missing keys, and nonfinite values. An entirely unobserved metric stays
`null`; observed zeros remain in the denominator. The existing metric sections
and names are preserved. Summary diagnostics add `stability_evaluated_samples`,
`stability_unevaluated_samples`, and `total_samples`. Chunk merging uses evaluated
sample counts for stability and sums coverage counts. Older summaries without
coverage counts retain their legacy chunk weights; previously inflated scores
cannot be corrected from old summaries alone and require reevaluation.

Run the regression tests without model dependencies or model loading:

```sh
python -m unittest discover -s tests -v
```
