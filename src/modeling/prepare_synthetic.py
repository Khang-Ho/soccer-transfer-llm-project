"""Create disjoint synthetic task-learning data; never use real QA answers."""

import argparse
import csv
import io
import json
import random
from pathlib import Path

from src.modeling.input_pipeline import CharacterTokenizer, FIELDS, canonical_query

ROOT = Path(__file__).resolve().parents[2]


def representations(v):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(v.keys())
    writer.writerow(v.values())
    nested = {
        "Player": {"Player ID": v["Player ID"], "Player Name": v["Player Name"],
                   "Personal Information": {k: v[k] for k in ("Date of Birth", "Country of Citizenship", "Height (cm)")},
                   "Career Information": {k: v[k] for k in ("Position", "Current Club")}},
        "Transfer": {"Transfer Date": v["Transfer Date"], "Transfer Season": v["Transfer Season"],
                     "Clubs": {k: v[k] for k in ("From Club", "To Club")},
                     "Financial Information": {k: v[k] for k in ("Transfer Fee", "Market Value (EUR)")}},
    }
    flat = (f"{v['Player Name']} has player ID {v['Player ID']}. "
            f"The player's date of birth is {v['Date of Birth']}, "
            f"country of citizenship is {v['Country of Citizenship']}, "
            f"height in cm is {v['Height (cm)']}, and position is {v['Position']}. "
            f"The player's current club is {v['Current Club']}. "
            f"On {v['Transfer Date']}, they transferred from {v['From Club']} "
            f"to {v['To Club']} in the {v['Transfer Season']} season. "
            f"The transfer fee is {v['Transfer Fee']}, "
            f"and the market value in EUR is {v['Market Value (EUR)']}.")
    return {"tabular": buffer.getvalue(), "key_value": json.dumps(v),
            "flat": flat, "hier_nested": json.dumps(nested)}


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def build_records(count, excluded_ids, seed=42):
    rng = random.Random(seed)
    records = []
    candidate = 900000000
    for i in range(count):
        while candidate in excluded_ids:
            candidate += 1
        records.append({
            "Player ID": candidate, "Player Name": f"Synthetic Player {i}",
            "Date of Birth": f"{rng.randrange(1980, 2006)}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}",
            "Country of Citizenship": rng.choice(['Testland A', 'Testland B', 'Testland C']),
            "Height (cm)": rng.randrange(160, 201),
            "Position": rng.choice(['Goalkeeper', 'Defender', 'Midfield', 'Attack']),
            "Current Club": f"Synthetic Club {rng.randrange(10,20)}",
            "Transfer Date": f"2020-{i // 28 % 12 + 1:02d}-{i % 28 + 1:02d}",
            "From Club": f"Synthetic Club {rng.randrange(10)}",
            "To Club": f"Synthetic Club {rng.randrange(10,20)}",
            "Transfer Season": "20/21", "Transfer Fee": rng.randrange(0, 101) * 10000,
            "Market Value (EUR)": rng.randrange(1, 101) * 10000,
        })
        candidate += 1
    return records


def make_examples(records, tokenizer):
    result = []
    for record in records:
        for field in FIELDS:
            item = {"player_id": record['Player ID'], "transfer_date": record['Transfer Date'], "target_field": field}
            query = canonical_query(item)
            answer = str(record[field])
            for structure, text in representations(record).items():
                for mode, prompt in (("context", text + '\n\n' + query), ("recall", query)):
                    tokenizer.encode(prompt + answer, strict=True)
                    result.append({**item, "structure": structure, "mode": mode,
                                   "prompt": prompt, "answer": answer})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-records', type=int, default=32)
    parser.add_argument('--validation-records', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.train_records, args.validation_records) < 1:
        parser.error('Both splits must contain records')
    with (ROOT / 'data/structures/key_value.jsonl').open(encoding='utf-8') as handle:
        excluded = {json.loads(line)['Player ID'] for line in handle}
    records = build_records(args.train_records + args.validation_records, excluded, args.seed)
    random.Random(args.seed).shuffle(records)
    tokenizer = CharacterTokenizer.load(ROOT / 'data/pilot/tokenizer.json')
    output = ROOT / 'data/synthetic'
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in [('train', records[:args.train_records]), ('validation', records[args.train_records:])]:
        examples = make_examples(split, tokenizer)
        write_jsonl(output / f'{name}.jsonl', examples)
        write_jsonl(output / f'{name}_records.jsonl', split)
        counts[name] = len(examples)
    metadata = {**vars(args), 'examples': counts, 'real_player_overlap': 0,
                'split_unit': 'synthetic player/record', 'tokenizer': 'data/pilot/tokenizer.json',
                'notes': 'All 4 structures balanced. Recall examples are repeated across structures to balance mode counts. Validation recall is an unseen-fact negative control, not expected to succeed. Context validation measures reading/query transfer.'}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
