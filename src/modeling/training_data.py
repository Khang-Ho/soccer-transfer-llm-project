"""Next-character corpus examples and answer-only task examples."""

import torch


def corpus_example(text, tokenizer, context_length):
    ids = tokenizer.encode(text, bos=True, eos=True, strict=True)
    if len(ids) > context_length:
        raise ValueError("Corpus record exceeds context length")
    return {"input_ids": ids[:-1], "labels": ids[1:]}


def answer_example(prompt, answer, tokenizer, context_length):
    prefix = tokenizer.encode(prompt, bos=True, strict=True)
    completion = tokenizer.encode(str(answer), eos=True, strict=True)
    ids = prefix + completion
    if len(ids) > context_length:
        raise ValueError("Query and answer exceed context length")
    labels = ids[1:]
    # First answer is predicted at the final prompt position.
    labels[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
    return {"input_ids": ids[:-1], "labels": labels}


def collate_examples(examples, pad_id=0):
    if not examples:
        raise ValueError("Empty batch")
    length = max(len(item["input_ids"]) for item in examples)
    inputs = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    for index, item in enumerate(examples):
        size = len(item["input_ids"])
        if not size or size != len(item["labels"]):
            raise ValueError("Invalid example length")
        inputs[index, :size] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[index, :size] = torch.tensor(item["labels"], dtype=torch.long)
    return {"input_ids": inputs, "labels": labels}
