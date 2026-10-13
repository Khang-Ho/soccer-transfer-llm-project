# Model and training design — initial pilot

Status: proposed pilot; no model has been trained yet.

## Research target

Measure how the representation of transfer facts affects recall in a small
decoder-only Transformer trained from random initialization. Compare tabular,
key-value, flat, and hierarchical nested representations of the same records.
Natural-language parsing and response formatting are separate deterministic
components, not capabilities assumed of the model.

## Query interface

Use existing evaluation metadata to construct a canonical query, without reading
the answer or injecting the source record:

    player_id=123|transfer_date=2024-07-01|field=transfer_fee|answer=

The model generates only the answer, ending with an explicit end marker.
English questions are a display layer. Initially support the known templates
or a form; arbitrary English parsing is outside the first pilot.
Only the scorer accesses expected answers. Formatting must not correct wrong
answers or invent units. Permit an explicit unknown answer.

## Initial architecture

- Character-level tokenizer, with fixed special tokens for padding and boundaries.
- Shared vocabulary built from training inputs only, plus fixed query syntax.
  Include an unknown token for characters absent from this vocabulary.
- Causal decoder-only Transformer: initially 2 layers, 4 attention heads,
  embedding width 128, feed-forward width 512, context limit 2048 characters.
- Count actual parameters after building the model; do not assume an exact size.
- Cross-entropy next-token loss; ignore padding. Use causal and padding masks.
- AdamW optimizer. Initial learning rate 0.001, batch size 4; these are pilot
  starting points, not established optimal settings or a memory guarantee.
- Check longest input before training; do not silently truncate records.

## Teaching the common query task

Corpus-only next-character training does not teach the model to answer the
canonical query. Conversely, converting every fact into identical QA examples
would erase the representation difference.

Proposed solution: first teach reading/querying using synthetic transfers,
whose IDs and facts are disjoint from the real evaluation records. Training
examples contain a represented synthetic record, the common query, and its
answer. Balance all four representations and use answer-token loss for this
stage. Include repeated synthetic records followed by query-only examples to
exercise recall without a record in the prompt.

Verify this task first on held-out synthetic examples. Copy the exact resulting
weights into four experimental models. Synthetic task training is shared;
no pretrained external weights or real evaluation QA pairs are used.

Then train each copy on only its real represented corpus using next-character
loss. Evaluate query-only recall, without supplying records or searching files.
Whether synthetic task learning transfers to real facts is an empirical pilot
question. If it fails, diagnose querying and memorization separately before
interpreting a format ranking. No guarantee is made that this design succeeds.

## Pilot data and existing evaluation file

Keep data/evaluation/recall_questions.jsonl unchanged as the draft full-scale
evaluation set. It already contains structured ID, date, field and answer data,
so English text does not need to be parsed for automated evaluation.

Start with a reproducible sample of 100 transfer records from the existing train
split. Choose pilot records independently of the frozen full evaluation set.
Use the same rows in all four formats. Build a separate pilot question file for
these records, excluding missing answers and ambiguous events. Keep full-scale
and pilot scores separate. Never test a pilot on all 1000 draft questions as if
their facts had all been trained.

## Fair comparison and reporting

- Same pilot rows, order, tokenizer, architecture, initial checkpoint and query
  interface; independently train four copies, not one sequentially adapted model.
- Primary comparison: equal passes over facts. Record tokens, updates, runtime
  and peak memory because representation lengths differ. A token-budget comparison
  can be reported separately and must not be conflated with equal exposure.
- Record seeds and deterministic settings; seed 42 is the initial pilot seed.
  Repeat successful experiments with additional seeds before robust conclusions.
- Use separate development questions to tune settings; freeze final evaluation
  before reporting. Do not tune repeatedly using the same final 1000 questions.
- Report exact-match accuracy overall and by field with documented conservative
  normalization. Store raw generations, failures and expected answers.
- Measure query-only recall before and after corpus training. Include a
  record-in-context diagnostic to distinguish reading failures from recall failures.
- Synthetic data must have no overlap with real evaluation identities/events.
- Existing validation/test transfers are unseen-fact sets, not the primary
  assessment of memorization of train facts. Report their role separately.

## Implementation order

1. Build tokenizer, canonical query conversion and pilot data preparation.
2. Implement Transformer, generation and scorer; verify masks and answer loss.
3. Check GPU availability, input lengths and memory with a short run.
4. Validate synthetic query-task learning and pilot memorization.
5. Run the four controlled pilot experiments; expand only after diagnostics pass.
