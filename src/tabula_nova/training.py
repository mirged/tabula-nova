"""Token-weighted AdamW training with deterministic sampling and FSDP2."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import PackedDataset, load_prepared_metadata, validate_tokenizer_json
from .model import TabulaNova
from .sampling import StepSampler


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 2048
    micro_batch_size: int = 1
    accumulation_steps: int = 32
    max_steps: int = 10000
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 100
    seed: int = 20261015

    def __post_init__(self):
        for name in ("seq_len", "micro_batch_size", "accumulation_steps", "max_steps", "eval_every", "eval_batches", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        if not 0 < self.min_lr <= self.peak_lr or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid optimizer configuration")


def learning_rate(step, config):
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps
    fraction = (step-config.warmup_steps) / max(1, config.max_steps-config.warmup_steps-1)
    return config.min_lr + (config.peak_lr-config.min_lr) * (1+math.cos(math.pi*min(1, fraction))) / 2


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("model_type") != "tabula_nova":
        raise ValueError("expected a Tabula Nova checkpoint (Tabula weights have different shapes)")
    return state


def resolve_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices: cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def inference_model(path, device="auto"):
    from tokenizers import Tokenizer
    from .data import validate_tokenizer

    device = resolve_device(device)
    state = load_checkpoint(path)
    config = ModelConfig(**state["model_config"])
    with torch.device("meta"):
        model = TabulaNova(config)
    model.load_state_dict(state["model"], assign=True)
    model.rope = type(model.rope)(config.head_dim, config.max_seq_len, config.rope_theta)
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    tokenizer = Tokenizer.from_str(state["tokenizer_json"])
    validate_tokenizer(tokenizer, config.vocab_size)
    return model, tokenizer


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, dataset, batch_size=1, device=None, max_batches=20, rank=0, world_size=1):
    device = device or next(model.parameters()).device
    # Equal forward counts on all ranks are required for FSDP collectives.
    batches = math.ceil(len(dataset)/(batch_size*world_size))
    batches = batches if max_batches is None else min(max_batches, batches)
    total = torch.zeros(2, device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        for batch in range(batches):
            rows = []
            for slot in range(batch_size):
                index = (batch*world_size+rank)*batch_size+slot
                if index < len(dataset):
                    rows.append(dataset[index])
                else:
                    x, y = dataset[0]
                    rows.append((x, torch.full_like(y, -100)))
            inputs, targets = (torch.stack([row[i] for row in rows]).to(device) for i in (0, 1))
            with autocast(device):
                loss = model(inputs, targets, loss_reduction="sum")
            total[0] += loss.double()
            total[1] += ((targets != -100) & (targets != model.config.pad_id)).sum()
        if world_size > 1:
            dist.all_reduce(total)
        if total[1].item() == 0:
            raise ValueError("evaluation has no supervised tokens")
        return (total[0]/total[1]).item()
    finally:
        model.train(was_training)


def train(data, output, model_config, config, device="auto", resume=None, fsdp=False):
    if config.seq_len > model_config.max_seq_len:
        raise ValueError("training sequence length exceeds model context")
    if model_config.parameter_count > 1_000_000_000 and not fsdp:
        raise ValueError("8B training requires --fsdp; use --preset tiny or pilot for local checks")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not fsdp:
        raise ValueError("torchrun with multiple ranks requires --fsdp")
    device = resolve_device(device)
    rank, world_size = 0, 1
    if fsdp:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("FSDP2 training requires BF16-capable CUDA GPUs")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group("nccl", device_id=device)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    elif device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this trainer requires BF16; use CPU for tiny checks")
    try:
        return _train(data, output, model_config, config, device, resume, fsdp, rank, world_size)
    finally:
        if fsdp and dist.is_initialized():
            dist.destroy_process_group()


def _train(data, output, model_config, config, device, resume, fsdp, rank, world_size):
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    data, output = Path(data), Path(output)
    meta = load_prepared_metadata(data)
    tokenizer_json = (data/"tokenizer.json").read_text()
    validate_tokenizer_json(tokenizer_json, model_config.vocab_size, meta["tokenizer_vocab_size"])
    dataset = PackedDataset(data/"train.bin", config.seq_len, model_config.pad_id)
    validation = PackedDataset(data/"validation.bin", config.seq_len, model_config.pad_id)
    torch.manual_seed(config.seed)
    if fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, register_fsdp_forward_method
        with torch.device("meta"):
            model = TabulaNova(model_config)
        mesh = init_device_mesh("cuda", (world_size,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            if model_config.residual_block_size:
                register_fsdp_forward_method(block, "forward_depth")
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        model = TabulaNova(model_config).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=config.peak_lr, betas=(0.9, 0.95), eps=1e-8,
                                 foreach=False)
    output.mkdir(parents=True, exist_ok=True)
    step, tokens_seen = 0, 0
    if resume:
        if fsdp:
            import torch.distributed.checkpoint as dcp
            saved = json.loads((Path(resume)/"training.json").read_text())
            model_state, optim_state = get_state_dict(model, optimizer)
            state = {"model": model_state, "optimizer": optim_state}
        else:
            saved = load_checkpoint(resume)
        if (saved["model_config"] != model_config.to_dict() or saved["train_config"] != asdict(config)
            or saved["data_sha256"] != meta["sha256"] or saved["world_size"] != world_size):
            raise ValueError("resume requires identical model, schedule, world size and data hashes")
        if fsdp:
            dcp.load(state, checkpoint_id=resume)
            set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
            rng = torch.load(Path(resume)/f"rng-{rank}.pt", weights_only=True)
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            rng = saved
        torch.set_rng_state(rng["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["rng_cuda"], device)
        step, tokens_seen = saved["step"], saved["tokens_seen"]

    def save():
        saved = {"format_version": 1, "model_type": "tabula_nova", "model_config": model_config.to_dict(),
                 "train_config": asdict(config), "data_sha256": meta["sha256"], "world_size": world_size,
                 "tokenizer_json": tokenizer_json, "step": step, "tokens_seen": tokens_seen}
        rng = {"rng_cpu": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        if fsdp:
            import torch.distributed.checkpoint as dcp
            destination = output/f"step-{step:08d}"
            destination.mkdir(exist_ok=True)
            m, o = get_state_dict(model, optimizer)
            dcp.save({"model": m, "optimizer": o}, checkpoint_id=destination)
            atomic_save(rng, destination/f"rng-{rank}.pt")
            dist.barrier()
            if rank == 0:
                (destination/"training.json").write_text(json.dumps(saved, indent=2)+"\n")
                (output/"latest.json").write_text(json.dumps({"checkpoint": destination.name})+"\n")
        else:
            atomic_save({**saved, **rng, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, output/"latest.pt")

    # A dedicated generator keeps DataLoader's base seed out of model RNG state.
    loader = DataLoader(dataset, batch_size=config.micro_batch_size,
                        sampler=StepSampler(len(dataset), config, rank, world_size, step),
                        generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    iterator = iter(loader)
    model.train()
    while step < config.max_steps:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        batches = [next(iterator) for _ in range(config.accumulation_steps)]
        count = torch.tensor(sum(int(((y != -100) & (y != model_config.pad_id)).sum()) for _, y in batches), device=device)
        if fsdp:
            dist.all_reduce(count)
        if count.item() == 0:
            raise ValueError("update has no supervised targets")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device):
                loss = model(inputs, targets, loss_reduction="sum")
            # FSDP averages rank gradients; undo that before token normalization.
            (loss * (world_size/count)).backward()
            loss_sum += loss.detach()
        if fsdp:
            dist.all_reduce(loss_sum)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        tokens_seen += int(count.item())
        metrics = {"step": step, "tokens_seen": tokens_seen, "loss": (loss_sum/count).item(),
                   "lr": optimizer.param_groups[0]["lr"], "grad_norm": norm.item(),
                   "supervised_tokens_per_second": count.item()/(time.perf_counter()-start)}
        if device.type == "cuda":
            metrics["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device)/2**30
        if step % config.eval_every == 0 or step == config.max_steps:
            metrics["validation_loss"] = evaluate(model, validation, config.micro_batch_size, device,
                                                   config.eval_batches, rank, world_size)
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with (output/"metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics)+"\n")
        if step % config.save_every == 0 or step == config.max_steps:
            save()
    return {"step": step, "tokens_seen": tokens_seen}
