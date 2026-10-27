# Query memorization diagnostic — October 27, 2026

## Question

Can the existing architecture learn and generate complete answers for a tiny
fixed set of supervised query-answer examples? This isolates basic optimization
and generation from record reading and larger task complexity.

## Fixed experiment

```powershell
.venv/Scripts/python.exe -m src.modeling.diagnose_learning --steps 500 --out checkpoints/overfit_four_queries_seed42
```

- First two distinct synthetic training players; height and transfer fee only.
- Four unique query-only QA pairs; repeated format labels are deduplicated.
- Fresh random initialization; seed 42, width 32, two layers, four heads,
  context 256, no dropout: 41,792 parameters. This differs from both the previous
  99,136-parameter smoke model and the default 691,456-parameter design.
- AdamW, learning rate 0.003, weight decay zero, full batch of four examples.
- Fixed 500 updates; diagnostics every 100 steps do not control stopping.
- Answer-token cross-entropy, greedy generation, exact answer and EOS required.
- Four queries from separate synthetic validation players are an unseen-fact
  negative control. No real football data or full evaluation questions are used.

## Observations

- Before training: 0/4 correct; token-weighted loss 5.15228.
- At checkpoints 100, 200, 300, 400, 500: 4/4 correct.
- Final token-weighted loss: 0.0009202.
- Unseen synthetic facts: 0/4 correct.
- Saved/reloaded checkpoint reproduces all four final generated answers.
- Config, selected QA examples, before/after outputs, unseen outputs, metrics,
  tokenizer, summary and checkpoint saved under the run directory.

This establishes tiny supervised memorization only. It does not establish reading
unseen records, transfer from corpus training to query answering, broad query
competence, superiority of any representation, or adequate capacity for real data.
The changes in training budget, context and dropout mean this is not a controlled
explanation of why the earlier 60-update mixed-task smoke run scored zero.

## Evaluation improvement

The mixed synthetic training runner now selects diagnostic examples per target
field as well as mode and representation. With one example per group and all
ten fields, each split has 40 context prompts and 10 unique recall prompts,
instead of the previous four context and four duplicated recall generations.
Old saved summaries retain their original meaning and are not rewritten.

## Next diagnostic

Test context reading separately using multiple synthetic records and all ten
fields, with per-field generations and genuinely unseen validation records.
Only after the querying diagnostics succeed should the shared initialization
and real four-representation memorization experiment proceed.
