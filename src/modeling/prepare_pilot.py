"""Run from the project root: python -m src.modeling.prepare_pilot."""

import csv
import hashlib
import io
import json
import random
import string
from collections import Counter
from pathlib import Path

from src.modeling.input_pipeline import CharacterTokenizer, FIELDS, canonical_query

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records), encoding="utf-8")


def flatten(record):
    result = {}
    for key, value in record.items():
        if isinstance(value, dict):
            result.update(flatten(value))
        else:
            result[key] = value
    return result


def main():
    source = ROOT / "data/structures/train"
    full_questions = ROOT / "data/evaluation/recall_questions.jsonl"
    original_hash = hashlib.sha256(full_questions.read_bytes()).hexdigest()
    reserved_players = {q["player_id"] for q in read_jsonl(full_questions)}
    records = read_jsonl(source / "key_value.jsonl")
    all_records = read_jsonl(ROOT / "data/structures/key_value.jsonl")
    counts = Counter((r["Player ID"], r["Transfer Date"]) for r in all_records)
    eligible = [i for i, r in enumerate(records)
                if r["Player ID"] not in reserved_players
                and counts[(r["Player ID"], r["Transfer Date"])] == 1
                and all(r[k] is not None and str(r[k]).strip() for k in ("Player ID", "Player Name", "Transfer Date", *FIELDS))]
    rng = random.Random(42)
    rng.shuffle(eligible)
    chosen, seen = [], set()
    for index in eligible:
        player = records[index]["Player ID"]
        if player not in seen:
            chosen.append(index)
            seen.add(player)
        if len(chosen) == 100:
            break
    if len(chosen) != 100:
        raise ValueError("Need 100 eligible distinct players")
    chosen.sort()
    selected = [records[i] for i in chosen]
    output = ROOT / "data/pilot"
    output.mkdir(parents=True, exist_ok=True)
    with (source / "tabular.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        table = list(reader)
    flat = read_jsonl(source / "flat.jsonl")
    nested = read_jsonl(source / "hier_nested.jsonl")
    assert len(table) == len(flat) == len(nested) == len(records)
    texts = {name: [] for name in ("tabular", "key_value", "flat", "hier_nested")}
    for i in chosen:
        record = records[i]
        assert flatten(nested[i]) == record
        assert table[i] == {k: "" if v is None else str(v) for k, v in record.items()}
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        writer.writerow([table[i][key] for key in header])
        texts["tabular"].append(buffer.getvalue())
        texts["key_value"].append(json.dumps(record, ensure_ascii=False))
        texts["flat"].append(flat[i]["text"])
        texts["hier_nested"].append(json.dumps(nested[i], ensure_ascii=False))
    queries = []
    # This set is explicitly for pilot development, not final reporting.
    for pilot_index, (index, record) in enumerate(zip(chosen, selected)):
        for field in FIELDS:
            item = {"question_id": f"pilot_{len(queries)+1:04d}", "source_split": "train",
                    "purpose": "pilot_development", "pilot_record_index": pilot_index,
                    "train_record_index": index, "player_id": record["Player ID"],
                    "transfer_date": record["Transfer Date"], "target_field": field}
            item["query"] = canonical_query(item)
            item["answer"] = record[field]
            queries.append(item)
    # No evaluation answers are inspected when constructing vocabulary.
    # Fixed ASCII supports the shared protocol and future synthetic examples.
    characters = set(string.printable)
    for corpus in texts.values():
        for text in corpus:
            characters.update(text)
    tokenizer = CharacterTokenizer(characters)
    lengths = {}
    for name, corpus in texts.items():
        encoded = [tokenizer.encode(text, bos=True, eos=True, strict=True) for text in corpus]
        lengths[name] = max(map(len, encoded))
        if lengths[name] > 2048:
            raise ValueError(f"{name} exceeds context limit; no truncation allowed")
        for text, ids in zip(corpus, encoded):
            assert tokenizer.decode(ids) == text
        write_jsonl(output / f"{name}.jsonl", [{"pilot_record_index": j, "train_record_index": i, "text": text} for j, (i, text) in enumerate(zip(chosen, corpus))])
    for q in queries:
        assert tokenizer.decode(tokenizer.encode(q["query"], strict=True)) == q["query"]
        assert q["answer"] == records[q["train_record_index"]][q["target_field"]]
    write_jsonl(output / "questions.jsonl", queries)
    tokenizer.save(output / "tokenizer.json")
    assert hashlib.sha256(full_questions.read_bytes()).hexdigest() == original_hash
    metadata = {"seed": 42, "records": 100, "questions": len(queries),
                "question_role": "development only; separate from full evaluation",
                "selection": "100 distinct players, complete fields, unambiguous events; excludes full-evaluation players",
                "train_record_indices": chosen, "index_base": 0,
                "vocab_size": tokenizer.vocab_size, "max_tokens_including_bos_eos": lengths,
                "context_limit": 2048, "tokenizer_source": "four pilot training corpora plus fixed ASCII protocol",
                "source_sha256": {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in ("tabular.csv", "key_value.jsonl", "flat.jsonl", "hier_nested.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"records_per_structure": 100, "pilot_questions": len(queries), "vocab_size": tokenizer.vocab_size, "max_tokens": lengths, "full_questions_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
