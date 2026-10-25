# Synthetic query-task training

Implemented two reproducible entry points from the project root:

```powershell
.venv/Scripts/python.exe -m src.modeling.prepare_synthetic --train-records 8 --validation-records 2
.venv/Scripts/python.exe -m src.modeling.train_synthetic --steps 60 --width 32 --batch-size 4 --max-new-tokens 32 --out checkpoints/synthetic_smoke_seed42
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Data generation uses seed 42, checks synthetic IDs against the full real dataset,
and separates train and validation by synthetic player/record. It keeps 13 fields
in all four representations. Generated data are in `data/synthetic`; real datasets,
pilot files and the frozen question file are not modified.

For each record, ask ten canonical queries in four representations, with two modes:

- Context: represented record + blank line + canonical query -> answer.
- Recall: canonical query without record -> answer.

Recall examples are identical across the four representation labels, deliberately
balancing the counts of the two modes. These are task-training examples, not
independent evidence for a representation comparison. The synthetic phase trains
one shared model across all formats. Real evaluation QA answers are not used.

All answers and their EOS are teacher-forced during training with loss only on
answer targets. Validation never updates weights. Train example order uses a
dedicated Python random generator; validation does not consume the training RNG.
Before/after losses are weighted by supervised token count, not batch count.
No special numeric normalization is used for generated exact-match scoring.

The runner saves arguments, model configuration, tokenizer, input hashes,
per-step metrics, token-weighted losses, raw predictions with stopping reasons,
and a checkpoint containing model/optimizer and sampler states. Reloaded weights
are checked to reproduce logits. Automatic resume is not implemented.
Output directories must be new to avoid replacing prior runs. Data preparation
can regenerate its files; checkpoints record their hashes.

## Initial CPU smoke test — October 25, 2026

- 8 synthetic train records, 2 validation records; 640/160 task examples.
- 60 optimizer updates, batch size 4, learning rate 0.001, seed 42.
- Reduced width 32: 99,136 parameters, not the default 691,456-parameter model.
- CPU PyTorch 2.14.1; no CUDA training performed.
- Train context loss: 5.0487 -> 3.1993.
- Train recall loss: 4.9307 -> 3.0559.
- Validation context loss: 5.0533 -> 3.2541.
- Validation recall loss: 4.9474 -> 3.1204.
- Diagnostic exact match: 0/4 in each split/mode group (16 generations total).
- Checkpoint round-trip passed. All 10 code tests passed.

The generation check takes the first example per structure/mode/split, so it is
very small and repetitive; it is not a balanced evaluation across target fields.
Unseen-record recall is a negative control, not a goal to infer unseen random facts.
Several fields still share values across records, so chance or learned marginal
predictions may occasionally be correct.

Interpretation: training and artifact saving work, but query competence has not
been demonstrated. Do not use this smoke checkpoint to initialize the real
four-format experiment. Next use broader per-field generation diagnostics and
small-task overfitting to separate optimization failures, task-format failures,
and reading versus memory failures. A decrease in teacher-forced loss alone is
insufficient evidence for either querying or football recall.
