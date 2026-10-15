"""Prepare a streaming text/code JSONL corpus using Tabula's digit-aware BPE."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile


def records(paths):
    from .synthetic import split_for
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError(f"{path}:{line_number}: expected nonempty text")
                # Never trust supplied split labels: identical content must stay
                # in the same split, including the tokenizer fitting sample.
                yield row["text"], split_for(row["text"])


def prepare_text(paths, output, vocab_size=32768, sample_documents=18000):
    import numpy as np
    from .data import train_tokenizer, file_hash, validate_tokenizer

    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    if sample_documents < 1:
        raise ValueError("sample_documents must be positive")
    if not 260 <= vocab_size <= 65535:
        raise ValueError("uint16 corpus requires vocabulary size 260..65535")
    def samples():
        count = 0
        for text, split in records(paths):
            if split == "train":
                yield text
                count += 1
                if count >= sample_documents:
                    break
        if count == 0:
            raise ValueError("no train-split tokenizer samples")
    tokenizer = train_tokenizer(samples(), vocab_size)
    validate_tokenizer(tokenizer, vocab_size)
    splits = ("train", "validation", "test")
    root.parent.mkdir(parents=True, exist_ok=True)
    counts, documents = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="nova-text-", dir=root.parent) as temporary:
        destination = Path(temporary)
        tokenizer.save(str(destination/"tokenizer.json"))
        with ExitStack() as stack:
            handles = {split: stack.enter_context((destination/f"{split}.bin").open("wb")) for split in splits}
            for text, split in records(paths):
                tokens = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
                handles[split].write(np.asarray(tokens, dtype="<u2").tobytes())
                counts[split] += len(tokens)
                documents[split] += 1
        if any(counts[split] < 2 for split in splits):
            raise ValueError("input must contain hashed examples in all three splits; add more diverse documents")
        artifacts = ["tokenizer.json", *[f"{split}.bin" for split in splits]]
        metadata = {"format_version": 1, "dtype": "<u2", "tokens": dict(counts),
                    "documents": dict(documents), "model_vocab_size": vocab_size,
                    "tokenizer_vocab_size": tokenizer.get_vocab_size(), "source_type": "jsonl_text",
                    "sha256": {name: file_hash(destination/name) for name in artifacts}}
        (destination/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        destination.rename(root)
    return metadata
