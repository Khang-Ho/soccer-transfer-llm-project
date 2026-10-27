"""Bounded synthetic training with token-weighted losses and raw generations."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.modeling.input_pipeline import CharacterTokenizer
from src.modeling.training_data import answer_example, collate_examples
from src.modeling.transformer import ModelConfig, RecallTransformer

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def select_diagnostics(rows, per_group):
    """Cover each target field; do not count identical recall prompts four times."""
    groups = {}
    seen = set()
    selected = []
    for row in rows:
        identity = (row['mode'], row['prompt'])
        if identity in seen:
            continue
        structure = row['structure'] if row['mode'] == 'context' else 'shared'
        key = (structure, row['mode'], row['target_field'])
        if groups.get(key, 0) < per_group:
            selected.append(row)
            seen.add(identity)
            groups[key] = groups.get(key, 0) + 1
    return selected


@torch.no_grad()
def evaluate_loss(model, examples, batch_size, device):
    previous = model.training
    model.eval()
    total, tokens = 0., 0
    try:
        for start in range(0, len(examples), batch_size):
            batch = {k: v.to(device) for k, v in collate_examples(examples[start:start+batch_size]).items()}
            n = int(batch['labels'].ne(-100).sum())
            total += float(model(**batch)['loss']) * n
            tokens += n
    finally:
        model.train(previous)
    return total / tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--eval-per-group', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads, args.eval_per_group, args.max_new_tokens) < 1 or args.lr <= 0:
        parser.error('Budgets and learning rate must be positive')
    if args.out.exists():
        parser.error('Choose a new output directory to avoid replacing a previous run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; no automatic CPU fallback')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)  # training order independent of evaluation/logging
    tokenizer = CharacterTokenizer.load(ROOT / 'data/pilot/tokenizer.json')
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=args.width,
                         feedforward_dim=args.width * 4)
    model = RecallTransformer(config).to(args.device)
    rows = {split: read_jsonl(ROOT / f'data/synthetic/{split}.jsonl') for split in ('train','validation')}
    filtered_train = []
    for row in rows["train"]:
        if row["mode"] == "context":
            filtered_train.append(row)
    rows["train"] = filtered_train
    filtered_validation = []
    for row in rows["validation"]:
        if row["mode"] == "context":
            filtered_validation.append(row)
    rows["validation"] = filtered_validation
    print("Train:", len(rows["train"]))
    print("Validation:", len(rows["validation"]))
    examples = {split: [answer_example(r['prompt'], r['answer'], tokenizer, config.context_length) for r in values] for split, values in rows.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True)
    tokenizer.save(args.out / 'tokenizer.json')
    manifest = {'arguments': {**vars(args), 'out': str(args.out)}, 'model': asdict(config),
                'parameters': model.parameter_count, 'torch_version': torch.__version__,
                'task': 'synthetic query training, not a four-format football comparison',
                'data_hashes': {s: hashlib.sha256((ROOT / f'data/synthetic/{s}.jsonl').read_bytes()).hexdigest() for s in rows}}
    (args.out / 'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    started = time.perf_counter()
    # Full loss evaluation is deterministic and token-weighted, separated by task.
    losses = {}
    for split in rows:
        for mode in ('context',):
            subset = [e for e,r in zip(examples[split], rows[split]) if r['mode']==mode]
            losses[f'{split}_{mode}_before'] = evaluate_loss(model, subset, args.batch_size, args.device)
    order, offset = [], 0
    with (args.out / 'metrics.jsonl').open('w', encoding='utf-8') as metrics:
        model.train()
        for step in range(1, args.steps+1):
            if offset >= len(order):
                order = list(range(len(examples['train'])))
                rng.shuffle(order)
                offset = 0
            indices = order[offset:offset+args.batch_size]
            offset += len(indices)
            batch = {k:v.to(args.device) for k,v in collate_examples([examples['train'][i] for i in indices]).items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            entry = {'step':step, 'batch_loss':float(loss.detach()), 'gradient_norm':float(norm), 'elapsed_seconds':time.perf_counter()-started}
            metrics.write(json.dumps(entry)+'\n')
            metrics.flush()
            if step == 1 or step % 20 == 0 or step == args.steps:
                print(json.dumps(entry), flush=True)
    for split in rows:
        for mode in ('context',):
            subset = [e for e,r in zip(examples[split],rows[split]) if r['mode']==mode]
            losses[f'{split}_{mode}_after'] = evaluate_loss(model, subset, args.batch_size, args.device)
    # Per-field diagnostic subset; shared recall prompts are counted only once.
    predictions = []
    for split, values in rows.items():
        for row in select_diagnostics(values, args.eval_per_group):
            result = model.generate(tokenizer.encode(row['prompt'], bos=True, strict=True), max_new_tokens=args.max_new_tokens)
            answer = tokenizer.decode(result['token_ids'])
            predictions.append({**row, 'split':split, 'prediction':answer,
                                'stop_reason':result['stop_reason'],
                                'correct':result['stop_reason']=='eos' and answer==row['answer']})
    (args.out/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions),encoding='utf-8')
    summary = {'losses':losses, 'diagnostic_exact_match':{
        f'{s}_{m}': {'correct':sum(p['correct'] for p in predictions if p['split']==s and p['mode']==m),
                    'total':sum(1 for p in predictions if p['split']==s and p['mode']==m)}
        for s in rows for m in ('context',)},
        'elapsed_seconds':time.perf_counter()-started,
        'limitation':'Tiny diagnostic subset; validation recall involves unseen facts. Loss decrease alone does not establish correct answers or football recall.'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    torch.save({'model_config':asdict(config), 'state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(), 'steps':args.steps,
                'torch_rng_state':torch.get_rng_state(), 'sampler_rng_state':rng.getstate(),
                'sampler_order':order,'sampler_offset':offset},args.out/'model.pth')
    # Check checkpoint round-trip against logits, without advancing training.
    saved = torch.load(args.out/'model.pth', map_location=args.device, weights_only=True)
    restored = RecallTransformer(ModelConfig(**saved['model_config'])).to(args.device).eval()
    restored.load_state_dict(saved['state_dict'])
    model.eval()
    check = {k:v.to(args.device) for k,v in collate_examples(examples['train'][:1]).items()}
    with torch.no_grad():
        torch.testing.assert_close(model(**check)['logits'],restored(**check)['logits'])
    print(json.dumps(summary,indent=2), flush=True)
    print('PASS: checkpoint reload reproduces logits.', flush=True)


if __name__ == '__main__':
    main()
