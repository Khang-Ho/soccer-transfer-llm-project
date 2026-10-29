# Context-only training experiment log

Use this file to track the development of the context-reading task. Append new
runs instead of replacing earlier results. Separate measured observations from
possible explanations. This is a development log, not the final comparison of
the four soccer data representations.

## Research question for this stage

Can a small Transformer answer a structured query when the relevant synthetic
transfer record is included in its input? This tests record reading and answer
generation, not query-only recall of real soccer facts.

## Shared setup for runs 1 and 2

- Data: 8 synthetic training players and 2 separate validation players.
- Context-only examples: 320 train and 80 validation, covering four representations
  and ten target fields. Each prompt contains one represented record and a query.
- Model: randomly initialized causal Transformer; 99,136 parameters, 2 layers,
  4 heads, width 32, feed-forward width 128, dropout 0.1, context limit 2048.
- Tokenizer: shared pilot character tokenizer, vocabulary size 127.
- Training: AdamW, learning rate 0.001, default weight decay 0.01, batch size 4,
  seed 42, CPU with 2 threads, PyTorch 2.14.1+cpu.
- Loss: answer-token cross-entropy with teacher forcing; prompt and padding targets
  are ignored. Reported dataset losses are weighted by supervised token count.
- Generation: greedy decoding, maximum 48 new tokens.
- Exact match requires the complete expected answer and EOS termination.
- Diagnostic subset: 40 train and 40 validation prompts (one per target field
  and representation), not all 320/80 examples. Related questions share records,
  so these are not 40 independent players or events.
- Both runs start from scratch. Run 2 increases the update budget, rather than
  continuing from run 1's checkpoint. Saved configurations and data hashes match
  except for steps and output directory.

Source JSONL hashes recorded in both configurations:

```text
train:      329dc4594de0772d50e497d886288c304015e7fb9c29bb7f37b731cf971275c4
validation: 810e49f1a7583e145f5b0c4d0551a1fd70a6171202ee6967faa043d1317de7cc
```

These hashes identify the original mixed-mode files; the training script filters
them to context-only examples before training.

## Results overview

| Run | Updates | Approx. passes over train examples | Train loss before -> after | Validation loss before -> after | Train exact match | Validation exact match | Runtime (s) |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 1 | 100 | 1.25 | 5.0487 -> 2.5142 | 5.0533 -> 2.5976 | 0/40 (0%) | 0/40 (0%) | 65.25 |
| 2 | 1,000 | 12.5 | 5.0487 -> 0.8461 | 5.0533 -> 1.4752 | 0/40 (0%) | 1/40 (2.5%) | 195.81 |

Runtime includes training and evaluation, not just optimizer updates. The passes
estimate is updates * batch size / 320.

## Run 1 — initial context-only test

Recorded: October 29, 2026. Configuration and values verified against saved files.

**Goal:** Check whether a short training run begins to learn the context-query task.

```powershell
.venv\Scripts\python.exe -m src.modeling.train_synthetic --steps 100 --width 32 --batch-size 4 --seed 42 --out checkpoints/context_only_run1
```

**Evidence:** `checkpoints/context_only_run1/config.json`, `summary.json`,
`metrics.jsonl`, `predictions.jsonl`, and `model.pth`.

**Observed:** Loss decreased on both splits, but no diagnostic answer matched.
For synthetic player 900000007, tabular questions about date of birth, height,
and transfer fee produced repetitive text such as `ethethe...000...` rather than
the expected values `1991-04-22`, `164`, and `590000`. The inspected validation
date-of-birth question showed the same pattern. These outputs reached
`max_new_tokens` rather than stopping with EOS. Checkpoint reload reproduced logits.

**Interpretation:** Lower teacher-forced loss did not translate into correct
free-running answers. The inspected failures are not merely minor formatting
differences. They suggest weak conditioning on the requested field and unstable
generation, but do not by themselves identify the underlying cause.

**Next decision:** Increase only the training budget to 1,000 updates while keeping
the other settings unchanged, to test whether more optimization helps.

## Run 2 — increased training budget

Recorded: October 29, 2026. Configuration and values verified against saved files.

**Goal:** Determine whether 10 times as many updates improve generation accuracy.

```powershell
.venv\Scripts\python.exe -m src.modeling.train_synthetic --steps 1000 --width 32 --batch-size 4 --seed 42 --out checkpoints/context_only_run2
```

**Evidence:** `checkpoints/context_only_run2/config.json`, `summary.json`,
`metrics.jsonl`, `predictions.jsonl`, and `model.pth`.

**Observed:** Train loss decreased to 0.8461 and validation loss to 1.4752.
Train exact match stayed at 0/40; validation exact match reached 1/40.
Checkpoint reload reproduced logits. Final train/validation loss gap grew from
about 0.0834 in run 1 to 0.6292 in run 2.

**Interpretation:** More updates improved the teacher-forced objective, but exact
answer generation remains poor. The larger loss gap warrants investigation of
generalization; it is not sufficient by itself to diagnose overfitting. One correct
validation answer does not establish reliable reading or generalization. Its field
and expected value must be inspected to assess whether a common answer could
explain the match. These results do not support ranking the four representations.

**Pending inspection before another training change:**

- Inspect the same date-of-birth, height and fee prompts as in run 1.
- Compare expected and generated answers and their stopping reasons.
- Identify the one correct validation answer and its field/representation.
- Count EOS versus token-budget termination and summarize failures by target field.
- Decide the next diagnostic from these observations, rather than changing several
  hyperparameters at once.

## Notes on reporting and preservation

- Existing `summary.json` files still contain an old limitation sentence referring
  to validation recall. These runs are context-only; interpret them accordingly.
- A successful checkpoint reload checks saving/loading, not answer correctness.
- The NumPy warning did not prevent these CPU runs from completing.
- `checkpoints/` is ignored by Git. Committing this Markdown file does not back up
  raw predictions or checkpoints. Preserve run folders separately when needed.
- Do not overwrite old run folders. Use a fresh name for each experiment.

## Template for future runs

Copy the section below for each new run and add a row to the overview table.

### Run N — descriptive title

- **Date:**
- **Question/hypothesis:**
- **Change from the previous run:**
- **Settings kept fixed:**
- **Command:**
- **Output folder and input/configuration references:**
- **Training and evaluation example counts:**
- **Train/validation loss before and after:**
- **Exact matches and evaluated sample counts:**
- **Representative predictions and stop reasons:**
- **Runtime and checkpoint verification:**
- **Observations (directly measured):**
- **Possible explanations (not yet proven):**
- **Limitations:**
- **Next action and reason:**
