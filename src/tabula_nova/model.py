"""Bias-free, pre-normalized causal decoder with headwise QK-RMSNorm."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .progress import progress, progress_enabled
from .config import ModelConfig, model_preset


def activation_checkpoint(function, *args):
    return checkpoint(function, *args, use_reentrant=False)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # FP32 reductions avoid overflow in FP16 at aggressive learning rates.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta):
        super().__init__()
        frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_seq_len).float(), frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        cos = self.cos[offset:offset+x.size(-2)].to(x.dtype)[None, None]
        sin = self.sin[offset:offset+x.size(-2)].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.dim // config.heads
        self.q = nn.Linear(config.dim, config.dim, bias=False)
        self.k = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.dim, self.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.output_gate = nn.Linear(config.dim, config.heads, bias=False) if config.attention_gate else None
        self.native_gqa = config.native_gqa

    def forward(self, x, rope, cache=None, use_cache=False):
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        offset = 0 if cache is None else cache[0].size(-2)
        q, k = rope(self.q_norm(q), offset), rope(self.k_norm(k), offset)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        new_cache = (k, v) if use_cache else None
        # The cache keeps KV heads compact. Native GQA is requested on CUDA;
        # CPU and the opt-out path use the explicit expansion reference.
        groups = self.heads // self.kv_heads
        native = self.native_gqa and x.device.type == "cuda" and groups > 1
        if groups > 1 and not native:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        mask = None
        if offset and length > 1:
            queries = torch.arange(offset, offset + length, device=x.device)
            keys = torch.arange(offset + length, device=x.device)
            mask = keys[None, :] <= queries[:, None]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            is_causal=offset == 0, dropout_p=0.0, enable_gqa=native)
        if self.output_gate is not None:
            gates = self.output_gate(x).sigmoid().transpose(1, 2).unsqueeze(-1)
            attended = attended * gates
        output = self.out(attended.transpose(1, 2).contiguous().view(batch, length, -1))
        return (output, new_cache) if use_cache else output


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.up = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.down = nn.Linear(config.mlp_dim, config.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DepthMix(nn.Module):
    """Token-local, content-dependent softmax over completed depth groups."""
    def __init__(self, config):
        super().__init__()
        self.key_norm = RMSNorm(config.dim, config.norm_eps)
        self.query = nn.Parameter(torch.zeros(config.dim))

    def forward(self, *representations):
        values = torch.stack(representations, dim=0)
        keys = self.key_norm(values).float()
        scores = (keys * self.query.float()).sum(-1)
        weights = scores.softmax(dim=0).to(values.dtype)
        return (values * weights.unsqueeze(-1)).sum(0)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.mlp_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.mlp = SwiGLU(config)
        if config.residual_block_size:
            self.attn_depth = DepthMix(config)
            self.mlp_depth = DepthMix(config)

    def forward(self, x, rope, cache=None, use_cache=False):
        attention = self.attn(self.attn_norm(x), rope, cache, use_cache)
        if use_cache:
            attention, new_cache = attention
        x = x + attention
        x = x + self.mlp(self.mlp_norm(x))
        return (x, new_cache) if use_cache else x

    def forward_depth(self, partial, rope, *completed, cache=None, use_cache=False):
        values = completed if partial is None else (*completed, partial)
        attention = self.attn(self.attn_norm(self.attn_depth(*values)), rope, cache, use_cache)
        if use_cache:
            attention, new_cache = attention
        attention = attention.to(completed[0].dtype)
        partial = attention if partial is None else partial + attention
        mlp = self.mlp(self.mlp_norm(self.mlp_depth(*completed, partial))).to(partial.dtype)
        partial = partial + mlp
        return (partial, new_cache) if use_cache else partial


class TabulaNova(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.embedding = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.rope = RotaryEmbedding(c.dim // c.heads, c.max_seq_len, c.rope_theta)
        self.final_depth = DepthMix(c) if c.residual_block_size else None
        self.reset_parameters()

    def reset_parameters(self):
        """Also usable after meta construction / FSDP2 to_empty()."""
        c = self.config
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, DepthMix):
                nn.init.zeros_(module.query)
        device = self.embedding.weight.device
        frequency = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=device).float() / c.head_dim))
        angles = torch.outer(torch.arange(c.max_seq_len, device=device).float(), frequency)
        self.rope.cos, self.rope.sin = angles.cos(), angles.sin()
        # Scale residual projections with depth for stable initialization.
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(block.mlp.down.weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, targets=None, *, past_key_values=None, use_cache=False,
                last_logits_only=False, loss_reduction="mean"):
        """Return logits, or mean CE for explicitly NEXT-token-shifted targets.

        targets has the same shape as input_ids; -100 and PAD targets are ignored.
        PAD supervision is ignored. Packed documents share causal context;
        document boundaries do not introduce a separate attention mask.
        """
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError(f"input_ids must be [batch, length], length 1..{self.config.max_seq_len}")
        if loss_reduction not in ("sum", "mean"):
            raise ValueError("loss_reduction must be sum or mean")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache")
        if use_cache and (self.training or targets is not None):
            raise ValueError("KV cache is for evaluation without targets")
        if past_key_values is not None:
            if len(past_key_values) != self.config.layers:
                raise ValueError("one cache entry is required per layer")
            offsets = {pair[0].size(-2) for pair in past_key_values}
            if len(offsets) != 1 or next(iter(offsets)) + input_ids.size(1) > self.config.max_seq_len:
                raise ValueError("cache lengths must agree and fit context")
            for key, value in past_key_values:
                if key.shape != value.shape or key.size(0) != input_ids.size(0) or key.size(1) != self.config.kv_heads or key.size(-1) != self.config.head_dim:
                    raise ValueError("invalid KV cache shape")
        x = self.embedding(input_ids)
        next_cache = []
        use_checkpoint = self.training and self.config.gradient_checkpointing and torch.is_grad_enabled()
        if self.config.residual_block_size:
            completed, partial = [x], None
            for index, block in enumerate(self.blocks):
                if index and index % self.config.residual_block_size == 0:
                    completed.append(partial)
                    partial = None
                arguments = (partial, self.rope, *completed)
                if use_cache:
                    cache = None if past_key_values is None else past_key_values[index]
                    partial, cached = block.forward_depth(*arguments, cache=cache, use_cache=True)
                    next_cache.append(cached)
                else:
                    partial = (activation_checkpoint(block.forward_depth, *arguments) if use_checkpoint
                               else block.forward_depth(*arguments))
            x = self.final_depth(*completed, partial)
        else:
            for index, block in enumerate(self.blocks):
                if use_cache:
                    cache = None if past_key_values is None else past_key_values[index]
                    x, cached = block(x, self.rope, cache, True)
                    next_cache.append(cached)
                else:
                    # Invoke __call__ so FSDP2 hooks run during recomputation too.
                    x = activation_checkpoint(block, x, self.rope) if use_checkpoint else block(x, self.rope)
        x = self.norm(x)
        # F.linear ties the output head directly to the embedding parameter.
        if targets is None:
            logits = F.linear(x[:, -1:] if last_logits_only else x, self.embedding.weight)
            return (logits, tuple(next_cache)) if use_cache else logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids and already be shifted one token")
        targets = targets.masked_fill(targets == self.config.pad_id, -100)
        hidden, flat_targets = x.flatten(0, 1), targets.flatten()
        size = self.config.loss_chunk_size or len(flat_targets)
        loss_sum = x.new_zeros((), dtype=torch.float32)
        for start in range(0, len(flat_targets), size):
            arguments = (hidden[start:start + size], self.embedding.weight, flat_targets[start:start + size])
            # Recompute each vocabulary projection on backward instead of retaining
            # the full B*T*V logits/log-softmax tensors. The tied weight remains
            # an explicit checkpoint input to preserve its gradient.
            loss_sum = loss_sum + (activation_checkpoint(self._loss_chunk, *arguments) if use_checkpoint
                                   else self._loss_chunk(*arguments))
        return loss_sum if loss_reduction == "sum" else loss_sum / (targets != -100).sum().clamp_min(1)

    @staticmethod
    def _loss_chunk(hidden, weight, targets):
        return F.cross_entropy(F.linear(hidden, weight).float(), targets,
                               ignore_index=-100, reduction="sum")

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.8, top_k=40,
                 allowed_vocab_size=None, show_progress=False, use_cache=True):
        """Single unpadded prompt; stop at EOS or the trained context limit."""
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 1:
            raise ValueError("generation expects one nonempty unpadded prompt")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("prompt exceeds the configured context")
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
            raise ValueError("invalid generation arguments")
        limit = self.config.vocab_size if allowed_vocab_size is None else allowed_vocab_size
        if not max(self.config.pad_id, self.config.bos_id, self.config.eos_id) < limit <= self.config.vocab_size:
            raise ValueError("allowed vocabulary must include special tokens and fit model")
        was_training = self.training
        self.eval()
        cache = None
        budget = min(max_new_tokens, self.config.max_seq_len - input_ids.size(1))
        try:
            for _ in progress(range(budget), desc="Generating", unit="token",
                              disable=not show_progress or not progress_enabled()):
                if use_cache:
                    current = input_ids if cache is None else input_ids[:, -1:]
                    logits, cache = self(current, past_key_values=cache, use_cache=True,
                                         last_logits_only=True)
                else:
                    logits = self(input_ids, last_logits_only=True)
                logits = logits[:, -1].float()
                logits[:, limit:] = -torch.inf
                logits[:, self.config.pad_id] = -torch.inf
                logits[:, self.config.bos_id] = -torch.inf
                if temperature == 0:
                    token = logits.argmax(-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k:
                        cutoff = logits.topk(min(top_k, limit), dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < cutoff, -torch.inf)
                    token = torch.multinomial(logits.softmax(-1), 1)
                input_ids = torch.cat((input_ids, token), dim=1)
                if token.item() == self.config.eos_id:
                    break
            return input_ids
        finally:
            self.train(was_training)
