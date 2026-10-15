"""Stream inference shards and build OCI-compatible ZIP64 model artifacts."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from .config import ModelConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_groups(shapes, max_bytes, bytes_per_element=2):
    if max_bytes < 1:
        raise ValueError("shard size must be positive")
    groups, current, size = [], [], 0
    for name, shape in shapes.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        cost = elements*bytes_per_element
        if cost > max_bytes:
            raise ValueError(f"{name} exceeds shard size; increase --shard-size-mib")
        if current and size+cost > max_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += cost
    if current:
        groups.append(current)
    return groups


def runtime_config(env_slug, env_path, python_version="3.11"):
    from .oci_storage import parse_uri
    parse_uri(env_path)
    if not env_slug or "\n" in env_slug:
        raise ValueError("provide the published inference conda environment slug")
    if python_version not in ("3.10", "3.11", "3.12", "3.13"):
        raise ValueError("unsupported Python version")
    # JSON is valid YAML; serialization avoids quoting / YAML injection issues.
    return {"MODEL_ARTIFACT_VERSION": "3.0", "MODEL_DEPLOYMENT": {"INFERENCE_CONDA_ENV": {
        "INFERENCE_ENV_SLUG": env_slug, "INFERENCE_ENV_TYPE": "published",
        "INFERENCE_ENV_PATH": env_path, "INFERENCE_PYTHON_VERSION": python_version}}}


def export_checkpoint(checkpoint, output, env_slug, env_path, python_version="3.11", shard_size_mib=1024):
    import torch
    from safetensors.torch import save_file
    from .model import TabulaNova
    from .data import validate_tokenizer_json

    runtime = runtime_config(env_slug, env_path, python_version)
    source, destination = Path(checkpoint), Path(output)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    if source.is_dir():
        import torch.distributed.checkpoint as dcp
        state = json.loads((source/"training.json").read_text())
        if state.get("model_type") != "tabula_nova":
            raise ValueError("expected Tabula Nova distributed checkpoint")
        weights = None
    else:
        # mmap avoids reading AdamW state into RAM just to export the model.
        state = torch.load(source, mmap=True, map_location="cpu", weights_only=True)
        if state.get("model_type") != "tabula_nova" or state.get("format_version") != 1:
            raise ValueError("expected Tabula Nova checkpoint")
        weights = state["model"]
    config = ModelConfig(**state["model_config"])
    vocab = json.loads(state["tokenizer_json"])["model"]["vocab"]
    validate_tokenizer_json(state["tokenizer_json"], config.vocab_size, len(vocab))
    with torch.device("meta"):
        reference = TabulaNova(config)
    shapes = {name: tuple(tensor.shape) for name, tensor in reference.state_dict().items()}
    groups = shard_groups(shapes, shard_size_mib*1024**2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nova-export-", dir=destination.parent) as temporary:
        root = Path(temporary)
        mapping = {}
        for index, names in enumerate(groups, 1):
            name = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            if weights is None:
                tensors = {key: torch.empty(shapes[key], dtype=torch.bfloat16) for key in names}
                # DCP supports partial state dictionaries. Load only this shard,
                # converting saved FP32 values to BF16 without loading AdamW.
                dcp.load({"model": tensors}, checkpoint_id=source)
            else:
                tensors = {key: weights[key].detach().to(dtype=torch.bfloat16).contiguous() for key in names}
            save_file(tensors, str(root/name), metadata={"format": "pt"})
            mapping.update({key: name for key in names})
            del tensors
        config.save(root/"config.json")
        (root/"tokenizer.json").write_text(state["tokenizer_json"])
        (root/"model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": config.parameter_count*2}, "weight_map": mapping}, indent=2)+"\n")
        (root/"runtime.yaml").write_text(json.dumps(runtime, indent=2)+"\n")
        package = root/"tabula_nova"
        package.mkdir()
        for filename in ("__init__.py", "config.py", "model.py", "progress.py", "data.py", "synthetic.py", "inference.py"):
            shutil.copyfile(Path(__file__).with_name(filename), package/filename)
        shutil.copyfile(Path(__file__).parent/"templates"/"score.py", root/"score.py")
        (root/"requirements.txt").write_text("torch>=2.8,<2.9\ntokenizers>=0.21,<0.23\nsafetensors>=0.5,<1\nnumpy>=1.24,<3\ntqdm>=4.66,<5\n")
        files = sorted(path for path in root.rglob("*") if path.is_file())
        manifest = {"format_version": 1, "model_type": "tabula_nova", "parameters": config.parameter_count,
                    "weight_dtype": "bfloat16", "training_step": state["step"],
                    "training_tokens": state["tokens_seen"], "architecture": config.to_dict(),
                    "sha256": {str(path.relative_to(root)): file_sha256(path) for path in files}}
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        root.rename(destination)
    return manifest


def zip_artifact(directory, output):
    root, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive output must be outside the artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
