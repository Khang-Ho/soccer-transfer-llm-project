"""Character tokenizer and answer-independent structured query interface."""

import json
from datetime import date
from pathlib import Path

FIELDS = {
    "Date of Birth": "date_of_birth",
    "Country of Citizenship": "country_of_citizenship",
    "Height (cm)": "height_in_cm",
    "Position": "position",
    "Current Club": "current_club",
    "From Club": "from_club",
    "To Club": "to_club",
    "Transfer Season": "transfer_season",
    "Transfer Fee": "transfer_fee",
    "Market Value (EUR)": "market_value_in_eur",
}


def canonical_query(item):
    player_id = item["player_id"]
    if isinstance(player_id, bool) or not isinstance(player_id, int) or player_id < 0:
        raise ValueError("player_id must be a nonnegative integer")
    transfer_date = item["transfer_date"]
    if date.fromisoformat(transfer_date).isoformat() != transfer_date:
        raise ValueError("Use an ISO YYYY-MM-DD date")
    field = FIELDS[item["target_field"]]
    return f"player_id={player_id}|transfer_date={transfer_date}|field={field}|answer="


class CharacterTokenizer:
    # Special tokens have dedicated IDs; literal text cannot inject them.
    SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")
    pad_id, bos_id, eos_id, unk_id = range(4)

    def __init__(self, characters):
        self.characters = sorted(set(characters))
        if any(len(char) != 1 for char in self.characters):
            raise ValueError("Vocabulary entries must be single characters")
        self.ids = {char: i + 4 for i, char in enumerate(self.characters)}

    @property
    def vocab_size(self):
        return len(self.characters) + 4

    def encode(self, text, *, bos=False, eos=False, strict=False):
        if strict and any(char not in self.ids for char in text):
            raise ValueError("Input contains a character absent from the vocabulary")
        tokens = [self.ids.get(char, self.unk_id) for char in text]
        return ([self.bos_id] if bos else []) + tokens + ([self.eos_id] if eos else [])

    def decode(self, tokens):
        result = []
        for token in tokens:
            if token == self.eos_id:
                break
            if token in (self.pad_id, self.bos_id):
                continue
            if token == self.unk_id:
                result.append("\ufffd")
            elif 4 <= token < self.vocab_size:
                result.append(self.characters[token - 4])
            else:
                raise ValueError(f"Invalid token ID: {token}")
        return "".join(result)

    def save(self, path):
        Path(path).write_text(json.dumps({"special_tokens": self.SPECIAL, "characters": self.characters}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["special_tokens"]) != cls.SPECIAL:
            raise ValueError("Incompatible special tokens")
        return cls(data["characters"])
