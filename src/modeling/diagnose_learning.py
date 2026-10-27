"""Small supervised recall overfit check, not a representation experiment."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.modeling.input_pipeline import CharacterTokenizer
from src.modeling.train_synthetic import ROOT, read_jsonl, evaluate_loss
from src.modeling.training_data import answer_example, collate_examples
from src.modeling.transformer import ModelConfig, RecallTransformer


def select_examples(rows, fields=("Height (cm)", "Transfer Fee"), players=2):
    selected = []
    seen = set()
    player_ids = []
    for row in rows:
        if row['mode'] != 'recall' or row['target_field'] not in fields:
            continue
        player = row['player_id']
        if player not in player_ids:
            if len(player_ids) == players:
                continue
            player_ids.append(player)
        key = row['prompt']
        if key not in seen:
            seen.add(key)
            selected.append(row)
    if len(selected) != players * len(fields):
        raise ValueError('Not enough complete player/field examples')
    return selected


@torch.no_grad()
def predict(model, tokenizer, rows):
    predictions = []
    for row in rows:
        result = model.generate(tokenizer.encode(row['prompt'], bos=True, strict=True), max_new_tokens=32)
        answer = tokenizer.decode(result['token_ids'])
        predictions.append({**row, 'prediction':answer, 'stop_reason':result['stop_reason'],
                            'correct':result['stop_reason']=='eos' and answer==row['answer']})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.out.exists():
        parser.error('Positive steps and a fresh output directory are required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tokenizer = CharacterTokenizer.load(ROOT/'data/pilot/tokenizer.json')
    rows = select_examples(read_jsonl(ROOT/'data/synthetic/train.jsonl'))
    heldout = select_examples(read_jsonl(ROOT/'data/synthetic/validation.jsonl'))
    assert not {r['player_id'] for r in rows} & {r['player_id'] for r in heldout}
    config = ModelConfig(vocab_size=tokenizer.vocab_size,context_length=256,
                         d_model=32,n_heads=4,n_layers=2,feedforward_dim=128,dropout=0)
    model = RecallTransformer(config)
    examples = [answer_example(r['prompt'],r['answer'],tokenizer,config.context_length) for r in rows]
    batch = collate_examples(examples)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
    args.out.mkdir(parents=True)
    tokenizer.save(args.out/'tokenizer.json')
    (args.out/'training_examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    before = predict(model,tokenizer,rows)
    initial_loss = evaluate_loss(model,examples,len(examples),'cpu')
    started=time.perf_counter()
    metrics=[]
    # Fixed update budget. Diagnostic outcomes do not control training or stopping.
    for step in range(1,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=model(**batch)['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==1 or step%100==0 or step==args.steps:
            generated=predict(model,tokenizer,rows)
            entry={'step':step,'batch_loss':float(loss.detach()),
                   'correct':sum(r['correct'] for r in generated),'total':len(rows)}
            metrics.append(entry)
            print(json.dumps(entry),flush=True)
    after=predict(model,tokenizer,rows)
    negative=predict(model,tokenizer,heldout)
    final_loss=evaluate_loss(model,examples,len(examples),'cpu')
    payload={'model_config':asdict(config),'state_dict':model.state_dict()}
    torch.save(payload,args.out/'model.pth')
    loaded=torch.load(args.out/'model.pth',weights_only=True)
    restored=RecallTransformer(ModelConfig(**loaded['model_config']))
    restored.load_state_dict(loaded['state_dict'])
    assert predict(restored,tokenizer,rows)==after
    for name,values in [('before',before),('after',after),('unseen_facts',negative),('metrics',metrics)]:
        (args.out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values),encoding='utf-8')
    summary={'seed':args.seed,'steps':args.steps,'model':asdict(config),
             'parameters':model.parameter_count,'torch_version':torch.__version__,
             'initial_loss':initial_loss,'final_loss':final_loss,
             'before_correct':sum(r['correct'] for r in before),
             'after_correct':sum(r['correct'] for r in after),'total':len(rows),
             'unseen_correct':sum(r['correct'] for r in negative),'unseen_total':len(negative),
             'elapsed_seconds':time.perf_counter()-started,'checkpoint_predictions_match':True,
             'limitation':'Supervised memorization of four QA pairs only. Does not establish record reading, unseen-fact inference, transfer from corpus training, or format superiority.'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
