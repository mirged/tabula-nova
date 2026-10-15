"""Streaming byte-corpus reconstruction, BPE preparation and packed datasets."""

from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .progress import progress, progress_enabled, progress_write

from .synthetic import (CATEGORY_WEIGHTS, TOPICS, TRAIN_TOPICS, DATASET_NAME, GENERATOR_VERSION,
                        document, generate_document, split_for)

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|sep|>"]
SPLITS = ("train", "validation", "test")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle, progress(total=Path(path).stat().st_size,
                                                  desc=f"Hash {Path(path).name}", unit='B',
                                                  unit_scale=True, leave=False) as bar:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            bar.update(len(block))
    return digest.hexdigest()


def legacy_documents(path):
    """Recover whole UTF-8 documents even when bytes straddle packed rows."""
    payload = None
    padded = False
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            ids = row["input_ids"]
            mask = row.get("attention_mask", [int(i != 0) for i in ids])
            if len(mask) != len(ids):
                raise ValueError(f"{path}:{line_number}: invalid mask length")
            for token, valid in zip(ids, mask):
                if type(token) is not int or not 0 <= token < 259 or valid not in (0, 1):
                    raise ValueError(f"{path}:{line_number}: invalid legacy byte token/mask")
                if token == 0 or valid == 0:
                    if token != 0 or valid != 0 or payload is not None:
                        raise ValueError(f"{path}:{line_number}: invalid padding")
                    padded = True
                    continue
                if padded:
                    raise ValueError(f"{path}:{line_number}: tokens follow padding")
                if token == 1:
                    if payload is not None:
                        raise ValueError(f"{path}:{line_number}: nested BOS")
                    payload = bytearray()
                elif token == 2:
                    if payload is None:
                        raise ValueError(f"{path}:{line_number}: EOS without BOS")
                    yield payload.decode("utf-8", errors="strict")
                    payload = None
                else:
                    if payload is None:
                        raise ValueError(f"{path}:{line_number}: bytes outside a document")
                    payload.append(token - 3)
    if payload is not None:
        raise ValueError(f"{path}: unfinished document")


def train_tokenizer(texts, vocab_size=8192):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if not 260 <= vocab_size <= 65536:
        raise ValueError("byte BPE vocabulary must be in 260..65536")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=SPECIAL_TOKENS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 show_progress=progress_enabled())
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def validate_tokenizer(tokenizer, model_vocab_size):
    if tokenizer.get_vocab_size() > model_vocab_size:
        raise ValueError("tokenizer exceeds model vocabulary")
    for index, name in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(name) != index:
            raise ValueError("tokenizer special token IDs must be PAD=0 BOS=1 EOS=2 SEP=3")
    validate_digit_tokenization(tokenizer)


def validate_digit_tokenization(tokenizer):
    samples = ("128", "1000", "-2048", "3/17", "value=500; next=10000")
    for text in samples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        offsets = getattr(encoded, "offsets", None)
        if offsets is None:
            raise ValueError("tokenizer must expose offsets to verify digit-level tokenization")
        for position, character in enumerate(text):
            if character.isdigit():
                covering = [(start, end) for start, end in offsets if start <= position < end]
                if len(covering) != 1 or covering[0] != (position, position + 1):
                    raise ValueError("digit-level tokenizer audit failed: a digit spans a merged token")


def validate_tokenizer_json(serialized, model_vocab_size, active_vocab_size):
    """Check saved BPE metadata without requiring tokenizers in the trainer."""
    spec = json.loads(serialized)
    if spec["model"]["type"] != "BPE":
        raise ValueError("prepared tokenizer must use BPE")
    vocabulary = dict(spec["model"]["vocab"])
    vocabulary.update({entry["content"]: entry["id"] for entry in spec.get("added_tokens", [])})
    if set(vocabulary.values()) != set(range(active_vocab_size)) or active_vocab_size > model_vocab_size:
        raise ValueError("saved tokenizer vocabulary does not match prepared metadata")
    for index, name in enumerate(SPECIAL_TOKENS):
        if vocabulary.get(name) != index:
            raise ValueError("saved tokenizer special token IDs are inconsistent")


class PackedJSONLWriter:
    """Stream fixed-length rows without losing boundary tokens or split tails."""
    def __init__(self, handle, sequence_length=512, include_loss_mask=False):
        if sequence_length <= 0:
            raise ValueError("packed sequence length must be positive")
        self.handle = handle
        self.sequence_length = sequence_length
        self.buffer = []
        self.loss_buffer = []
        self.include_loss_mask = include_loss_mask
        self.sequences = 0

    def write(self, ids):
        self.buffer.extend(ids)
        offset = 0
        while len(self.buffer) - offset >= self.sequence_length:
            self._emit({"input_ids": self.buffer[offset:offset + self.sequence_length]})
            offset += self.sequence_length
        if offset:
            del self.buffer[:offset]

    def write_document(self, ids, loss_mask=None):
        """Keep each training example in one row; pad between documents."""
        if len(ids) > self.sequence_length:
            raise ValueError("document exceeds packed sequence length")
        if len(self.buffer) + len(ids) > self.sequence_length:
            padding = self.sequence_length - len(self.buffer)
            record = {"input_ids": self.buffer + [0] * padding,
                      "attention_mask": [1] * len(self.buffer) + [0] * padding}
            if self.include_loss_mask:
                record["loss_mask"] = self.loss_buffer + [0] * padding
            self._emit(record)
            self.buffer.clear()
            self.loss_buffer.clear()
        self.buffer.extend(ids)
        if self.include_loss_mask:
            if loss_mask is None:
                loss_mask = [1] * len(ids)
            if len(loss_mask) != len(ids):
                raise ValueError("document loss mask does not align with tokens")
            self.loss_buffer.extend(loss_mask)

    def _emit(self, record):
        self.handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.sequences += 1

    def finish(self):
        if self.buffer:
            valid = len(self.buffer)
            padding = self.sequence_length - valid
            record = {"input_ids": self.buffer + [0] * padding,
                      "attention_mask": [1] * valid + [0] * padding}
            if self.include_loss_mask:
                record["loss_mask"] = self.loss_buffer + [0] * padding
            self._emit(record)
            self.buffer.clear()
            self.loss_buffer.clear()


def _loss_mask(text, encoded):
    mask = bytearray([1] * (len(encoded.ids) + 2))
    offsets = getattr(encoded, "offsets", None)
    if offsets is None:
        return mask
    spans, seen_symbols = [], set()
    for match in re.finditer(r"\bnode_[a-z]{8}\b", text):
        if match.group() not in seen_symbols:
            spans.append(match.span())
            seen_symbols.add(match.group())
    setup_end = text.find("Question:")
    if setup_end >= 0:
        spans.extend(m.span() for m in re.finditer(r"(?<![A-Za-z_])[-+]?\d+(?:/\d+)?(?![A-Za-z_])", text[:setup_end]))
    for token_index, (start, end) in enumerate(offsets, 1):
        if any(start < stop and end > begin for begin, stop in spans):
            mask[token_index] = 0
    return mask


def _write_document_to_loss(handle, tokenizer, text):
    handle.write(_loss_mask(text, tokenizer.encode(text, add_special_tokens=False)))


def _write_length_histogram(path, histograms, sequence_length):
    topics = sorted(histograms)
    width, row_height = 900, 62
    height = 50 + row_height * max(1, len(topics))
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<style>text{font:12px sans-serif;fill:#222}.bar{fill:#3769a5}.grid{stroke:#bbb;stroke-width:1}</style>',
              '<text x="12" y="20">Accepted document lengths by topic (32-token bins)</text>']
    plot_x, plot_width = 230, 640
    for tick in (0, sequence_length // 4, sequence_length // 2, 3 * sequence_length // 4, sequence_length):
        x = plot_x + plot_width * tick / sequence_length
        pieces.append(f'<line class="grid" x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="{height - 15}"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height - 2}" text-anchor="middle">{tick}</text>')
    for row, topic in enumerate(topics):
        y = 37 + row * row_height
        bins = histograms[topic]
        maximum = max(bins.values(), default=1)
        pieces.append(f'<text x="8" y="{y + 20}">{topic}</text>')
        for start_text, count in bins.items():
            start = int(start_text)
            bar_x = plot_x + plot_width * start / sequence_length
            bar_w = max(1, plot_width * 32 / sequence_length - 1)
            bar_h = 30 * count / maximum
            pieces.append(f'<rect class="bar" x="{bar_x:.1f}" y="{y + 44 - bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}"><title>{topic}: {start}–{start + 31} tokens, {count} documents</title></rect>')
    pieces.append('</svg>')
    Path(path).write_text('\n'.join(pieces), encoding='utf-8')


def _write_document(handle, tokenizer, text, packed_writer=None, whole_row=True, loss_handle=None,
                    mask_random_fields=True):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [1, *encoded.ids, 2]
    handle.write(np.asarray(ids, dtype="<u2").tobytes())
    if loss_handle is not None:
        loss_handle.write(_loss_mask(text, encoded) if mask_random_fields else bytes([1] * len(ids)))
    if packed_writer is not None:
        if whole_row:
            packed_writer.write_document(ids, _loss_mask(text, encoded) if packed_writer.include_loss_mask else None)
        else:
            packed_writer.write(ids)
    return len(ids)


def prepare_data(output, source=None, target_tokens=4_500_000_000, seed=20261013,
                 vocab_size=8192, tokenizer_sample_docs=18000, packed_seq_len=512):
    """Produce uint16 streams. source=None generates the spec taxonomy.

    Existing corpus splits stay intact. Synthetic category ratios are controlled
    by token deficits, with deterministic content-hash train/validation/test splits.
    Output is only published after preparation succeeds; existing files are refused.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new prepared directory")
    if target_tokens <= 0 or tokenizer_sample_docs <= 0 or packed_seq_len <= 0:
        raise ValueError("token target and tokenizer sample size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tabula-prepare-", dir=output.parent) as temp:
        root = Path(temp)
        sample_total = tokenizer_sample_docs
        if source is not None:
            source = Path(source)
            meta = json.loads((source / "metadata.json").read_text())
            if meta.get("vocab_size") != 259 or meta.get("special_tokens") != {"pad": 0, "bos": 1, "eos": 2}:
                raise ValueError("source must be the legacy 259-token byte corpus")
            samples = itertools.islice(legacy_documents(source / "train.jsonl"), tokenizer_sample_docs)
            document_counts = meta.get('documents', {})
            available = document_counts.get('train') if isinstance(document_counts, dict) else None
            sample_total = min(tokenizer_sample_docs, available) if isinstance(available, int) else None
        else:
            def samples_iter():
                rng = random.Random(seed)
                found = 0
                while found < tokenizer_sample_docs:
                    cat = rng.choices(list(CATEGORY_WEIGHTS), list(CATEGORY_WEIGHTS.values()))[0]
                    text = document(rng, cat, profile="train")
                    if split_for(text) == "train":
                        found += 1
                        yield text
            samples = samples_iter()
        with progress(samples, total=sample_total, desc='Tokenizer samples', unit='doc') as sample_bar:
            tokenizer = train_tokenizer(sample_bar, vocab_size)
        validate_tokenizer(tokenizer, vocab_size)
        tokenizer.save(str(root / "tokenizer.json"))
        tokens, docs, categories, topics = Counter(), Counter(), Counter(), Counter()
        storage_tokens = Counter()
        with ExitStack() as stack:
            handles = {s: stack.enter_context((root / f"{s}.bin").open("wb")) for s in SPLITS}
            loss_handles = {s: stack.enter_context((root / f"{s}.lossmask.bin").open("wb")) for s in SPLITS}
            packed = {s: PackedJSONLWriter(stack.enter_context((root / f"{s}.jsonl").open("w", encoding="utf-8")),
                                          packed_seq_len, include_loss_mask=(source is None)) for s in SPLITS}
            binary_rows, mask_rows = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}

            def flush_binary(split):
                if not binary_rows[split]:
                    return
                padding = packed_seq_len - len(binary_rows[split])
                handles[split].write(np.asarray(binary_rows[split] + [0] * padding, dtype="<u2").tobytes())
                loss_handles[split].write(bytes(mask_rows[split] + [0] * padding))
                storage_tokens[split] += packed_seq_len
                binary_rows[split].clear()
                mask_rows[split].clear()

            length_histogram = defaultdict(Counter)
            accepted_length_histogram = defaultdict(Counter)
            rejected_oversize = Counter()
            oversize_streak = Counter()
            if source is not None:
                for split in SPLITS:
                    for text in progress(legacy_documents(source / f"{split}.jsonl"),
                                         desc=f'Convert {split}', unit='doc'):
                        tokens[split] += _write_document(handles[split], tokenizer, text, packed[split], whole_row=False, loss_handle=loss_handles[split], mask_random_fields=False)
                        docs[split] += 1
                    progress_write(f"{split}: {tokens[split]:,} BPE tokens, {docs[split]:,} documents")
            else:
                rng = random.Random(seed + 1)
                next_report = 1_000_000
                total = 0
                # Small smoke corpora still need both held-out partitions.
                # Stop on complete documents, so token targets are lower bounds.
                with progress(total=target_tokens, desc='Generate and pack', unit='token', unit_scale=True) as corpus_bar:
                    while total < target_tokens or any(tokens[s] < 2 for s in SPLITS):
                        # Greedy deficits track desired TOKEN ratios, not document ratios.
                        eligible_categories = [c for c in CATEGORY_WEIGHTS if oversize_streak[c] < 100]
                        if not eligible_categories:
                            raise ValueError("no generated training documents fit the packed sequence length; increase --packed-seq-len")
                        category = max(eligible_categories,
                                       key=lambda c: CATEGORY_WEIGHTS[c] * total - categories[c])
                        generated = generate_document(rng, category, profile="train")
                        text = generated.text
                        ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
                        length_histogram[f"{category}/{generated.topic}"][str(min(len(ids) // 32 * 32, packed_seq_len))] += 1
                        if len(ids) > packed_seq_len:
                            rejected_oversize[f"{category}/{generated.topic}"] += 1
                            oversize_streak[category] += 1
                            continue
                        oversize_streak.clear()
                        accepted_length_histogram[f"{category}/{generated.topic}"][str(min(len(ids) // 32 * 32, packed_seq_len))] += 1
                        split = split_for(text)
                        count = len(ids)
                        doc_mask = list(_loss_mask(text, tokenizer.encode(text, add_special_tokens=False)))
                        if len(binary_rows[split]) + len(ids) > packed_seq_len:
                            flush_binary(split)
                        binary_rows[split].extend(ids)
                        mask_rows[split].extend(doc_mask)
                        packed[split].write_document(ids, doc_mask)
                        tokens[split] += count
                        docs[split] += 1
                        categories[category] += count
                        topics[f"{category}/{generated.topic}"] += 1
                        total += count
                        if corpus_bar.update(count):
                            corpus_bar.set_postfix(docs=sum(docs.values()),
                                                   rejected=sum(rejected_oversize.values()),
                                                   domain=category, refresh=False)
                        if total >= next_report:
                            if corpus_bar.disable:
                                print(f"Prepared {total:,}/{target_tokens:,} BPE tokens", flush=True)
                            next_report += 1_000_000
                for split in SPLITS:
                    flush_binary(split)
            for writer in packed.values():
                writer.finish()
        _write_length_histogram(root / "topic-lengths.svg", accepted_length_histogram,
                                packed_seq_len)
        if any(tokens[s] < 2 for s in SPLITS):
            raise ValueError("each split needs at least two tokens; increase the generation target")
        files = {f"{s}.{extension}": file_hash(root / f"{s}.{extension}")
                 for s in SPLITS for extension in ("bin", "jsonl", "lossmask.bin")}
        files["tokenizer.json"] = file_hash(root / "tokenizer.json")
        files["topic-lengths.svg"] = file_hash(root / "topic-lengths.svg")
        metadata = {"format_version": 2, "dtype": "<u2", "seed": seed,
                    "dataset": DATASET_NAME if source is None else str(meta.get("project", "legacy")) + "-BPE",
                    "generator_version": GENERATOR_VERSION if source is None else None,
                    "delimiter_format": "setup + question <|sep|> answer" if source is None else "source-dependent",
                    "packed_sequence_length": packed_seq_len,
                    "sequences": {s: packed[s].sequences for s in SPLITS},
                    "model_vocab_size": vocab_size, "tokenizer_vocab_size": tokenizer.get_vocab_size(),
                    "special_tokens": dict(zip(("pad", "bos", "eos", "sep"), range(4))),
                    "tokens": dict(tokens), "documents": dict(docs),
                    "storage_tokens": dict(storage_tokens) if source is None else dict(tokens),
                    "category_tokens": dict(categories), "category_weights": CATEGORY_WEIGHTS if source is None else None,
                    "topic_documents": dict(topics),
                    "length_histogram_32_token_bins": {k: dict(v) for k, v in length_histogram.items()} if source is None else {},
                    "accepted_length_histogram_32_token_bins": {k: dict(v) for k, v in accepted_length_histogram.items()} if source is None else {},
                    "rejected_oversize_documents": dict(rejected_oversize) if source is None else {},
                    "training_document_policy": "whole documents only; documents above packed_sequence_length are rejected",
                    "target_scale": "4.5B default (100x previous target); preparation may take substantial time",
                    "number_token_policy": "Digits(individual_digits=True) before ByteLevel BPE",
                    "training_topics": {k: list(v) for k, v in TRAIN_TOPICS.items()} if source is None else None,
                    "held_out_topic_families": {k: [t for t in TOPICS[k] if t not in TRAIN_TOPICS[k]] for k in TRAIN_TOPICS} if source is None else None,
                    "source": str(source) if source is not None else "procedural-v3",
                    "target_tokens": target_tokens if source is None else None,
                    "sha256": files, "elapsed_seconds": time.perf_counter() - start}
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.move(str(root), str(output))
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


class PackedDataset(Dataset):
    """Each item learns every adjacent token pair once, including split tails."""
    def __init__(self, path, seq_len=512, pad_id=0):
        self.path = Path(path)
        if seq_len < 1 or self.path.stat().st_size % 2:
            raise ValueError("invalid sequence length or truncated uint16 file")
        self.tokens = np.memmap(self.path, mode="r", dtype="<u2")
        mask_path = self.path.with_name(self.path.stem + ".lossmask.bin")
        self.loss_mask = np.memmap(mask_path, mode="r", dtype="u1") if mask_path.exists() else None
        if self.loss_mask is not None and len(self.loss_mask) != len(self.tokens):
            raise ValueError("loss mask must align with token stream")
        if len(self.tokens) < 2:
            raise ValueError("a split needs at least two tokens")
        self.seq_len, self.pad_id = seq_len, pad_id

    def __len__(self):
        return math.ceil((len(self.tokens) - 1) / self.seq_len)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_len
        segment = np.array(self.tokens[start:start + self.seq_len + 1], dtype=np.int64)
        inputs = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        targets = torch.full((self.seq_len,), -100, dtype=torch.long)
        count = len(segment) - 1
        inputs[:count] = torch.from_numpy(segment[:-1])
        targets[:count] = torch.from_numpy(segment[1:])
        targets[targets == self.pad_id] = -100
        targets[inputs == self.pad_id] = -100
        if self.loss_mask is not None:
            mask = np.array(self.loss_mask[start + 1:start + count + 1], dtype=np.uint8)
            targets[:count][torch.from_numpy(mask == 0)] = -100
        return inputs, targets


def load_prepared_metadata(directory, verify=True):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["format_version"] not in (1, 2) or metadata["dtype"] != "<u2":
        raise ValueError("unsupported prepared data format")
    for split in SPLITS:
        expected_tokens = metadata.get("storage_tokens", metadata["tokens"])[split]
        if (directory / f"{split}.bin").stat().st_size != expected_tokens * 2:
            raise ValueError(f"{split} token count does not match metadata")
        mask_path = directory / f"{split}.lossmask.bin"
        if mask_path.exists() and mask_path.stat().st_size != expected_tokens:
            raise ValueError(f"{split} loss mask does not match stored tokens")
    if verify:
        for name, expected in metadata["sha256"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"prepared artifact changed: {name}")
    return metadata
