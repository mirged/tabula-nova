"""Load an exported artifact without optimizer state or a training checkout."""

import hashlib
import json
from pathlib import Path


def load_artifact(path, device="auto", verify=True):
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from .config import ModelConfig
    from .data import validate_tokenizer
    from .model import TabulaNova, RotaryEmbedding

    root = Path(path)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("model_type") != "tabula_nova" or manifest.get("format_version") != 1:
        raise ValueError("unsupported artifact")
    if verify:
        for name, expected in manifest["sha256"].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("artifact path escapes root")
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"artifact checksum mismatch: {name}")
    config = ModelConfig.load(root/"config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type not in ("cuda", "cpu") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU or CUDA device")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.device("meta"):
        model = TabulaNova(config)
    expected_keys = set(model.state_dict())
    seen = set()
    index = json.loads((root/"model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        if Path(name).name != name:
            raise ValueError("invalid weight shard filename")
        tensors = load_file(str(root/name))
        if seen.intersection(tensors) or set(tensors)-expected_keys:
            raise ValueError("duplicate or unexpected weights")
        if any(index["weight_map"].get(key) != name for key in tensors):
            raise ValueError("weight shard does not match index")
        # Transfer one shard at a time so host RAM need not hold all GPU weights.
        tensors = {key: value.to(device=device, dtype=dtype) for key, value in tensors.items()}
        model.load_state_dict(tensors, strict=False, assign=True)
        seen.update(tensors)
        del tensors
    if seen != expected_keys or set(index["weight_map"]) != seen:
        raise ValueError("artifact is missing model weights")
    model.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
    model.to(device=device, dtype=dtype).eval()
    tokenizer = Tokenizer.from_file(str(root/"tokenizer.json"))
    validate_tokenizer(tokenizer, config.vocab_size)
    return model, tokenizer
