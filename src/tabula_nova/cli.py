"""Commands with lazy heavyweight imports so inspect/pack work without torch."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .config import ModelConfig, model_preset, budget


def add_model_options(parser):
    parser.add_argument("--preset", choices=("8b", "8b-baseline", "8b-attnres", "pilot", "tiny"), default="8b")
    parser.add_argument("--config", help="model configuration JSON; overrides preset")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tabula Nova 8B research and OCI tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="exact architecture and memory/compute estimates")
    add_model_options(inspect)
    inspect.add_argument("--world-size", type=int, default=8)
    inspect.add_argument("--tokens", type=int, default=160_000_000_000)
    inspect.add_argument("--verify-meta", action="store_true", help="instantiate on meta device to cross-check the parameter formula; requires torch")
    prepare = commands.add_parser("prepare", help="prepare the inherited verified procedural corpus")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--target-tokens", type=int, default=1_000_000)
    prepare.add_argument("--vocab-size", type=int, default=32768)
    prepare.add_argument("--tokenizer-sample-docs", type=int, default=18000)
    prepare.add_argument("--packed-seq-len", type=int, default=2048)
    prepare.add_argument("--seed", type=int, default=20261015)
    text = commands.add_parser("prepare-text", help="stream text/code JSONL and fit BPE on train content only")
    text.add_argument("--input", nargs="+", required=True)
    text.add_argument("--output", required=True)
    text.add_argument("--vocab-size", type=int, default=32768)
    text.add_argument("--tokenizer-sample-docs", type=int, default=18000)
    training = commands.add_parser("train")
    add_model_options(training)
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--train-config", help="training settings JSON; flags override")
    training.add_argument("--device", default="auto")
    training.add_argument("--resume")
    training.add_argument("--fsdp", action="store_true")
    for name in ("seq-len", "micro-batch-size", "accumulation-steps", "max-steps", "warmup-steps", "eval-every", "eval-batches", "save-every", "seed"):
        training.add_argument("--"+name, type=int)
    for name in ("peak-lr", "min-lr", "weight-decay", "grad-clip"):
        training.add_argument("--"+name, type=float)
    for name in ("generate", "benchmark", "evaluate"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--checkpoint")
        source.add_argument("--artifact")
        command.add_argument("--device", default="auto")
        if name == "generate":
            command.add_argument("--prompt", required=True)
            command.add_argument("--max-new-tokens", type=int, default=128)
            command.add_argument("--temperature", type=float, default=0)
            command.add_argument("--top-k", type=int, default=40)
        elif name == "benchmark":
            command.add_argument("--documents-per-domain", type=int, default=8)
            command.add_argument("--mode", choices=("answer", "trace"), default="answer")
            command.add_argument("--suite", choices=("iid", "template", "depth", "length"), default="iid")
            command.add_argument("--context-limit", type=int, default=1024)
            command.add_argument("--max-new-tokens", type=int, default=128)
        else:
            command.add_argument("--data", required=True)
            command.add_argument("--split", choices=("validation", "test"), default="test")
            command.add_argument("--seq-len", type=int, default=2048)
            command.add_argument("--max-batches", type=int, default=20)
    export = commands.add_parser("export-oci", help="export BF16 safetensors, score.py and runtime.yaml")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--env-slug", required=True)
    export.add_argument("--env-path", required=True)
    export.add_argument("--python-version", default="3.11")
    export.add_argument("--shard-size-mib", type=int, default=1024)
    export.add_argument("--zip", help="optional ZIP64 output path outside artifact directory")
    pack = commands.add_parser("pack-job", help="build source-only OCI job ZIP")
    pack.add_argument("--output", default="dist/tabula_nova_8b.zip")
    for name in ("upload", "download"):
        transfer = commands.add_parser(name)
        transfer.add_argument("--uri", required=True)
        transfer.add_argument("--path", required=True)
        transfer.add_argument("--auth", choices=("resource_principal", "instance_principal", "config"), default="resource_principal")
        transfer.add_argument("--profile", default="DEFAULT")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        config = ModelConfig.load(args.config) if args.config else model_preset(args.preset)
        result = {"config": config.to_dict(), **budget(config, world_size=args.world_size, tokens=args.tokens)}
        if args.verify_meta:
            import torch
            from .model import TabulaNova
            with torch.device("meta"):
                count = TabulaNova(config).parameter_count
            if count != config.parameter_count:
                raise RuntimeError(f"parameter formula mismatch: {count}")
            result["meta_verified"] = True
    elif args.command == "pack-job":
        from .oci_job import pack_job
        result = {"archive": str(pack_job(args.output))}
    elif args.command in ("upload", "download"):
        from . import oci_storage
        result = (oci_storage.upload(args.path, args.uri, args.auth, args.profile) if args.command == "upload"
                  else oci_storage.download(args.uri, args.path, args.auth, args.profile))
    elif args.command == "prepare":
        from .data import prepare_data
        result = prepare_data(args.output, target_tokens=args.target_tokens, seed=args.seed,
                              vocab_size=args.vocab_size, tokenizer_sample_docs=args.tokenizer_sample_docs,
                              packed_seq_len=args.packed_seq_len)
    elif args.command == "prepare-text":
        from .corpus import prepare_text
        result = prepare_text(args.input, args.output, args.vocab_size, args.tokenizer_sample_docs)
    elif args.command == "train":
        from .training import train, TrainConfig
        config = ModelConfig.load(args.config) if args.config else model_preset(args.preset)
        values = json.loads(Path(args.train_config).read_text()) if args.train_config else {}
        for field in fields(TrainConfig):
            value = getattr(args, field.name)
            if value is not None:
                values[field.name] = value
        if args.preset == "tiny" and not args.config:
            values.setdefault("seq_len", 64)
        result = train(args.data, args.output, config, TrainConfig(**values), args.device, args.resume, args.fsdp)
    elif args.command == "export-oci":
        from .export import export_checkpoint, zip_artifact
        result = export_checkpoint(args.checkpoint, args.output, args.env_slug, args.env_path,
                                   args.python_version, args.shard_size_mib)
        if args.zip:
            result["archive"] = str(zip_artifact(args.output, args.zip))
    else:
        import torch
        if args.artifact:
            from .inference import load_artifact
            model, tokenizer = load_artifact(args.artifact, args.device)
        else:
            from .training import inference_model
            model, tokenizer = inference_model(args.checkpoint, args.device)
        if args.command == "generate":
            ids = [model.config.bos_id, *tokenizer.encode(args.prompt, add_special_tokens=False).ids]
            output = model.generate(torch.tensor([ids], device=next(model.parameters()).device),
                                    args.max_new_tokens, args.temperature, args.top_k, tokenizer.get_vocab_size())
            result = {"text": tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)}
        elif args.command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(model, tokenizer, args.documents_per_domain, mode=args.mode,
                               suite=args.suite, context_limit=args.context_limit, max_new_tokens=args.max_new_tokens)
        else:
            from .training import evaluate
            from .data import PackedDataset, load_prepared_metadata
            meta = load_prepared_metadata(args.data)
            if (Path(args.data)/"tokenizer.json").read_text() != tokenizer.to_str():
                # Compare parsed JSON because serialization whitespace can differ.
                if json.loads((Path(args.data)/"tokenizer.json").read_text()) != json.loads(tokenizer.to_str()):
                    raise ValueError("evaluation tokenizer differs from the model tokenizer")
            if args.seq_len > model.config.max_seq_len:
                raise ValueError("evaluation sequence length exceeds model context")
            dataset = PackedDataset(Path(args.data)/f"{args.split}.bin", args.seq_len, model.config.pad_id)
            result = {"split": args.split, "loss": evaluate(model, dataset, max_batches=args.max_batches)}
    print(json.dumps(result, indent=2))
