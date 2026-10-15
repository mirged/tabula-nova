"""Bounded pilot data collection and inspectable workspace stages.

Collection preserves raw records; cleaning records normalized content hashes.
The SQLite deduplication index bounds RAM use. Mixing uses a seeded shuffle
buffer, so it remains streaming and does not promise an exact token mixture.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile
import unicodedata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def output_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; select a new version directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def collect_web(root, documents=1000, seed=42, output=None):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as parquet

    if documents < 1:
        raise ValueError("document limit must be positive")
    registry = json.loads((root/"data/sources.json").read_text())
    source = next(item for item in registry["sources"] if item["id"] == "fineweb-edu")
    destination = output_directory(output or root/"data/raw/web-v1")
    api = HfApi()
    revision = api.dataset_info(source["dataset"]).sha
    prefix = source.get("file_prefix", "sample/10BT/")
    files = sorted(name for name in api.list_repo_files(source["dataset"], repo_type="dataset", revision=revision)
                   if name.startswith(prefix) and name.endswith(".parquet"))
    if not files:
        raise ValueError(f"no Parquet shards under {prefix}")
    rng = random.Random(seed)
    rng.shuffle(files)
    used_files = []
    cache = root/"data/cache/huggingface/hub"
    with tempfile.TemporaryDirectory(prefix="nova-web-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        records = stage/"documents.jsonl"
        count, buffer = 0, []
        with records.open("w", encoding="utf-8") as handle:
            def emit(row):
                handle.write(json.dumps({"text": row[source["text_field"]], "source": source["id"],
                    "source_id": row.get("id"), "url": row.get("url"),
                    "dataset_revision": revision, "dataset_license_label": source["dataset_license_label"]},
                    ensure_ascii=False)+"\n")
            for filename in files:
                # Cache one whole shard at a time, then read local batches with
                # threading disabled. This avoids the Arrow/fsspec shutdown
                # race seen with partially consumed remote Parquet streams.
                local = hf_hub_download(source["dataset"], filename, repo_type="dataset",
                                        revision=revision, cache_dir=str(cache))
                used_files.append({"file": filename, "sha256": sha256(local),
                                   "bytes": Path(local).stat().st_size})
                reader = parquet.ParquetFile(local)
                try:
                    for batch in reader.iter_batches(batch_size=512, columns=[source["text_field"], "id", "url"], use_threads=False):
                        for row in batch.to_pylist():
                            text = row.get(source["text_field"])
                            if not isinstance(text, str) or not text.strip():
                                continue
                            if len(buffer) < 128:
                                buffer.append(row)
                                continue
                            index = rng.randrange(len(buffer))
                            emit(buffer[index])
                            buffer[index] = row
                            count += 1
                            if count == documents:
                                break
                        if count == documents:
                            break
                finally:
                    reader.close()
                if count == documents:
                    break
            if count < documents:
                rng.shuffle(buffer)
                for row in buffer[:documents-count]:
                    emit(row)
                    count += 1
        if count != documents:
            raise ValueError(f"requested {documents} documents but source yielded {count}")
        manifest = {"stage": "collection", "created_utc": now(), "source": source,
                    "resolved_revision": revision, "documents": count, "seed": seed,
                    "sha256": {records.name: sha256(records)}, "downloaded_shards": used_files,
                    "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
                    "pyarrow_version": importlib.metadata.version("pyarrow"),
                    "reader": {"mode": "cached_local_parquet", "shuffle_buffer": 128,
                               "batch_size": 512, "use_threads": False}}
        write_json(stage/"manifest.json", manifest)
        stage.rename(destination)
    return manifest


def generate_synthetic(root, documents=1000, seed=20261015, output=None):
    from .synthetic import CATEGORY_WEIGHTS, GENERATOR_VERSION, generate_document

    if documents < 1:
        raise ValueError("document limit must be positive")
    destination = output_directory(output or root/"data/raw/synthetic-v1")
    rng = random.Random(seed)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="nova-synthetic-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        records = stage/"documents.jsonl"
        with records.open("w", encoding="utf-8") as handle:
            for _ in range(documents):
                domain = rng.choices(list(CATEGORY_WEIGHTS), list(CATEGORY_WEIGHTS.values()))[0]
                document = generate_document(rng, domain, profile="train")
                handle.write(json.dumps({"text": document.text, "source": "tabula-synth-v3",
                    "domain": domain, "topic": document.topic, "generator_version": GENERATOR_VERSION},
                    ensure_ascii=False)+"\n")
                counts[domain] += 1
        manifest = {"stage": "generation", "created_utc": now(), "seed": seed,
                    "documents": documents, "domains": dict(counts), "generator_version": GENERATOR_VERSION,
                    "profile": "train", "sha256": {records.name: sha256(records)}}
        write_json(stage/"manifest.json", manifest)
        stage.rename(destination)
    return manifest


def clean(inputs, output, min_chars=80, max_chars=100_000):
    from .synthetic import split_for

    if not 1 <= min_chars <= max_chars:
        raise ValueError("invalid character length bounds")
    destination = output_directory(output)
    counts, source_counts, splits = Counter(), Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="nova-clean-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        database = sqlite3.connect(stage/"dedup.sqlite")
        database.execute("CREATE TABLE hashes (digest TEXT PRIMARY KEY)")
        accepted, rejected = stage/"documents.jsonl", stage/"rejected.jsonl"
        try:
            with accepted.open("w", encoding="utf-8") as good, rejected.open("w", encoding="utf-8") as bad:
                for path in inputs:
                    with Path(path).open(encoding="utf-8") as handle:
                        for number, line in enumerate(handle, 1):
                            counts["seen"] += 1
                            try:
                                record = json.loads(line)
                                text = record["text"]
                                if not isinstance(text, str):
                                    raise ValueError("text must be a string")
                                text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
                                if not min_chars <= len(text) <= max_chars:
                                    raise ValueError("length")
                                if "\x00" in text:
                                    raise ValueError("nul_character")
                                # A conservative repetition check, not a semantic
                                # quality score; preserve repeated proof/code lines.
                                lines = [line.strip() for line in text.splitlines() if line.strip()]
                                if len(lines) >= 20 and len(set(lines))/len(lines) < 0.1:
                                    raise ValueError("repeated_lines")
                                digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                                inserted = database.execute("INSERT OR IGNORE INTO hashes VALUES (?)", (digest,)).rowcount
                                if not inserted:
                                    raise ValueError("duplicate")
                                record.update(text=text, content_sha256=digest)
                                source_counts[record.get("source", "unknown")] += 1
                                splits[split_for(text)] += 1
                                good.write(json.dumps(record, ensure_ascii=False)+"\n")
                                counts["accepted"] += 1
                            except (ValueError, KeyError, TypeError) as error:
                                reason = str(error) if str(error) in ("length", "nul_character", "repeated_lines", "duplicate") else "invalid_record"
                                counts["rejected_"+reason] += 1
                                bad.write(json.dumps({"input": str(path), "line": number, "reason": reason})+"\n")
                database.commit()
        finally:
            database.close()
        if not counts["accepted"]:
            raise ValueError("cleaning accepted no documents")
        manifest = {"stage": "cleaning", "created_utc": now(), "counts": dict(counts),
                    "sources": dict(source_counts), "content_hash_splits": dict(splits),
                    "inputs": {str(path): sha256(path) for path in inputs},
                    "filters": {"min_chars": min_chars, "max_chars": max_chars,
                                "deduplication": "exact normalized whitespace/NFC across all inputs"},
                    "sha256": {accepted.name: sha256(accepted)},
                    "pending": ["near-duplicate detection", "benchmark contamination checks", "PII/secret filtering", "grouped source splits"]}
        write_json(stage/"manifest.json", manifest)
        stage.rename(destination)
    return manifest


def mix(input_path, output, seed=42, buffer_size=512):
    if buffer_size < 1:
        raise ValueError("shuffle buffer must be positive")
    destination = output_directory(output)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="nova-mix-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        records = stage/"documents.jsonl"
        buffer, count = [], 0
        with Path(input_path).open(encoding="utf-8") as handle, records.open("w", encoding="utf-8") as result:
            for line in handle:
                if len(buffer) < buffer_size:
                    buffer.append(line)
                    continue
                index = rng.randrange(len(buffer))
                result.write(buffer[index])
                buffer[index] = line
                count += 1
            rng.shuffle(buffer)
            result.writelines(buffer)
            count += len(buffer)
        if not count:
            raise ValueError("mixture is empty")
        manifest = {"stage": "mixing", "created_utc": now(), "seed": seed, "buffer_size": buffer_size,
                    "documents": count, "input_sha256": sha256(input_path),
                    "sha256": {records.name: sha256(records)},
                    "note": "Preserves source counts; shuffles with bounded memory. Does not balance by tokens."}
        write_json(stage/"manifest.json", manifest)
        stage.rename(destination)
    return manifest


def doctor(root):
    import platform
    import sys
    packages = {}
    for name in ("torch", "numpy", "tokenizers", "safetensors", "datasets", "oci"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"checked_utc": now(), "python": sys.version.split()[0], "executable": sys.executable,
              "platform": platform.platform(), "free_disk_gib": shutil.disk_usage(root).free/2**30,
              "packages": packages, "cuda_available": False}
    if packages["torch"]:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["gpus"] = [{"name": torch.cuda.get_device_name(index),
                           "total_memory_gib": torch.cuda.get_device_properties(index).total_memory/2**30}
                          for index in range(torch.cuda.device_count())]
    (root/"reports").mkdir(exist_ok=True)
    write_json(root/"reports/environment.json", report)
    return report


def status(root):
    paths = {
        "environment": "reports/environment.json", "web_sample": "data/raw/web-v1/manifest.json",
        "synthetic_sample": "data/raw/synthetic-v1/manifest.json", "cleaned": "data/cleaned/pilot-v1/manifest.json",
        "mixed": "data/mixed/pilot-v1/manifest.json", "smoke_data": "prepared/workflow-smoke/metadata.json",
        "smoke_checkpoint": "runs/workflow-smoke/latest.pt", "native_tests": "reports/native-tests.json",
        "pilot_data": "prepared/workflow-pilot/metadata.json", "pilot_checkpoint": "runs/workflow-pilot/latest.pt",
        "oci_preflight": "reports/oci-preflight.json"}
    return {name: {"present": (root/path).is_file(), "path": path} for name, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "status", "collect", "synthetic", "clean", "mix"):
        command = commands.add_parser(name)
        if name in ("collect", "synthetic"):
            command.add_argument("--documents", type=int, default=1000)
        if name in ("collect", "synthetic"):
            command.add_argument("--output", type=Path)
        if name == "clean":
            command.add_argument("--input", nargs="+", type=Path)
            command.add_argument("--output", type=Path)
        if name == "mix":
            command.add_argument("--input", type=Path)
            command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "doctor":
        result = doctor(root)
    elif args.command == "status":
        result = status(root)
    elif args.command == "collect":
        result = collect_web(root, args.documents, output=args.output)
    elif args.command == "synthetic":
        result = generate_synthetic(root, args.documents, output=args.output)
    elif args.command == "clean":
        inputs = args.input or [root/"data/raw/web-v1/documents.jsonl", root/"data/raw/synthetic-v1/documents.jsonl"]
        result = clean(inputs, args.output or root/"data/cleaned/pilot-v1")
    else:
        result = mix(args.input or root/"data/cleaned/pilot-v1/documents.jsonl", args.output or root/"data/mixed/pilot-v1")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
