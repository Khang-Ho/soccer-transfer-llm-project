"""Build factual recall questions about records seen during training."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SEED = 42
PER_TYPE = 100
QUESTIONS = {
    "Date of Birth": "What is the player's date of birth? Answer in YYYY-MM-DD format.",
    "Country of Citizenship": "What is the player's country of citizenship?",
    "Height (cm)": "What is the player's height in centimetres? Answer with a number.",
    "Position": "What is the player's position according to the dataset?",
    "Current Club": "What is the player's current club according to the dataset snapshot?",
    "From Club": "Which club did the player transfer from in this transfer?",
    "To Club": "Which club did the player transfer to in this transfer?",
    "Transfer Season": "Which season does this transfer belong to?",
    "Transfer Fee": "What is the recorded transfer fee? Answer with a number in the dataset's original units.",
    "Market Value (EUR)": "What is the recorded market value for this transfer in EUR? Answer with a number.",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def context_key(record):
    return record["Player ID"], record["Transfer Date"]


def present(value):
    return value is not None and str(value).strip() != ""


def main():
    source = PROJECT / "data/structures/train/key_value.jsonl"
    full_source = PROJECT / "data/structures/key_value.jsonl"
    train = load_jsonl(source)
    full = load_jsonl(full_source)
    contexts = Counter(context_key(record) for record in full)
    rng = random.Random(SEED)
    selected = []
    used_players = set()
    for field, template in QUESTIONS.items():
        candidates = [
            index for index, record in enumerate(train)
            if contexts[context_key(record)] == 1
            and all(present(record[key]) for key in ("Player ID", "Player Name", "Transfer Date", field))
        ]
        rng.shuffle(candidates)
        picked = 0
        for index in candidates:
            record = train[index]
            if record["Player ID"] in used_players:
                continue
            selected.append({
                "question_id": f"recall_{len(selected) + 1:04d}",
                "source_split": "train",
                "train_record_index": index,
                "player_id": record["Player ID"],
                "transfer_date": record["Transfer Date"],
                "target_field": field,
                "question": (
                    f"According to the dataset, consider {record['Player Name']} "
                    f"(player ID {record['Player ID']}) and their transfer on "
                    f"{record['Transfer Date']}. {template}"
                ),
                "answer": record[field],
            })
            used_players.add(record["Player ID"])
            picked += 1
            if picked == PER_TYPE:
                break
        if picked != PER_TYPE:
            raise ValueError(f"Insufficient eligible players for {field}: {picked}")
    rng.shuffle(selected)
    assert len(selected) == 1000
    assert len({item["question"] for item in selected}) == 1000
    for item in selected:
        record = train[item["train_record_index"]]
        assert item["answer"] == record[item["target_field"]]
        assert contexts[context_key(record)] == 1
        assert present(item["answer"])
    output = PROJECT / "data/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "recall_questions.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in selected), encoding="utf-8")
    metadata = {
        "purpose": "Recall of facts provided during fine-tuning; shared across all four structures.",
        "source_split": "train",
        "seed": SEED,
        "question_count": len(selected),
        "counts_by_field": dict(Counter(item["target_field"] for item in selected)),
        "unique_players": len(used_players),
        "record_index_base": 0,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "100 questions per field; distinct players; player ID and transfer date uniquely identify the record in the full dataset; missing answers excluded.",
        "usage": "Evaluation only. Do not add questions or answers to training inputs. Freeze this set before comparing models; it is not evaluation of unseen transfer facts.",
    }
    (output / "recall_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    restored = load_jsonl(path)
    assert restored == selected
    print(f"PASS: {len(restored)} questions, 100 per type, 1000 distinct players; answers match train records; contexts unique in full dataset.")
    print(path)


if __name__ == "__main__":
    main()
