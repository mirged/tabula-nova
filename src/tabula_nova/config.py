"""Architecture and budgeting; intentionally usable without PyTorch."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 4096
    layers: int = 32
    heads: int = 32
    kv_heads: int = 8
    mlp_dim: int = 16640
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    norm_eps: float = 1e-6
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    gradient_checkpointing: bool = True
    attention_gate: bool = True
    residual_block_size: int = 0
    loss_chunk_size: int = 128
    native_gqa: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "dim", "layers", "heads", "kv_heads", "mlp_dim", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("dim must divide into heads; heads must divide into KV groups")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not all(math.isfinite(x) and x > 0 for x in (self.norm_eps, self.rope_theta)):
            raise ValueError("norm_eps and rope_theta must be finite and positive")
        if any(not isinstance(x, int) or x < 0 for x in (self.residual_block_size, self.loss_chunk_size)):
            raise ValueError("residual_block_size and loss_chunk_size must be nonnegative integers")
        if len({self.pad_id, self.bos_id, self.eos_id}) != 3 or any(
            not 0 <= i < self.vocab_size for i in (self.pad_id, self.bos_id, self.eos_id)
        ):
            raise ValueError("special token IDs must be distinct and in the vocabulary")

    @property
    def head_dim(self):
        return self.dim // self.heads

    @property
    def parameter_count(self):
        d = self.dim
        block = 2*d*d + 2*d*self.kv_heads*self.head_dim + 3*d*self.mlp_dim
        block += 2*d + 2*self.head_dim  # branch norms and headwise QK norms
        block += d*self.heads if self.attention_gate else 0
        block += 4*d if self.residual_block_size else 0
        return self.vocab_size*d + self.layers*block + d + (2*d if self.residual_block_size else 0)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def model_preset(name="8b", **overrides):
    if name == "8b":
        config = ModelConfig()
    elif name == "8b-baseline":
        config = ModelConfig(attention_gate=False)
    elif name == "8b-attnres":
        config = ModelConfig(residual_block_size=4)
    elif name == "pilot":
        config = ModelConfig(dim=576, layers=28, heads=9, kv_heads=3, mlp_dim=1536,
                             max_seq_len=1024, vocab_size=8192, rope_theta=10000.0)
    elif name == "tiny":
        config = ModelConfig(vocab_size=512, dim=64, layers=2, heads=4, kv_heads=2,
                             mlp_dim=192, max_seq_len=128, loss_chunk_size=16)
    else:
        raise ValueError(f"unknown model preset: {name}")
    return replace(config, **overrides)


def budget(config, *, world_size=8, tokens=160_000_000_000, batch_size=1, context=None):
    if world_size < 1 or tokens < 1 or batch_size < 1:
        raise ValueError("world size, tokens and batch size must be positive")
    context = context or config.max_seq_len
    if not 1 <= context <= config.max_seq_len:
        raise ValueError("context must fit the model")
    n = config.parameter_count
    return {
        "parameters": n, "target_billions": n / 1e9,
        "bf16_weights_gib": n * 2 / 2**30,
        "fp32_adamw_state_and_gradient_gib": n * 16 / 2**30,
        "sharded_persistent_state_gib_per_rank": n * 16 / world_size / 2**30,
        "kv_cache_gib": 2 * config.layers * batch_size * context * config.kv_heads * config.head_dim * 2 / 2**30,
        "training_tokens": tokens, "tokens_per_parameter": tokens / n,
        "approximate_training_flops_6NT": 6 * n * tokens,
        "note": "Planning estimates exclude activations, all-gathers, temporary buffers, attention cost and checkpoint I/O; measure GPU memory in a pilot.",
    }
