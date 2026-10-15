"""OCI Data Science scoring interface; included at the artifact ZIP root."""

import json
from pathlib import Path
import threading

_LOCK = threading.Lock()


def load_model(model_file_name=None):
    from tabula_nova.inference import load_artifact
    return load_artifact(Path(__file__).resolve().parent)


def predict(data, model=None):
    import torch
    if model is None:
        raise ValueError("OCI scoring server must pass the value returned by load_model")
    if isinstance(data, (bytes, str)):
        data = json.loads(data)
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), str) or not data["prompt"]:
        raise ValueError("provide a nonempty prompt string")
    network, tokenizer = model
    max_new_tokens = data.get("max_new_tokens", 128)
    temperature, top_k = data.get("temperature", 0.0), data.get("top_k", 40)
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be 1..1024")
    if not isinstance(temperature, (int, float)) or not isinstance(top_k, int):
        raise ValueError("temperature must be numeric and top_k must be an integer")
    ids = [network.config.bos_id, *tokenizer.encode(data["prompt"], add_special_tokens=False).ids]
    if len(ids)+max_new_tokens > network.config.max_seq_len:
        raise ValueError("prompt plus generation budget exceeds model context")
    inputs = torch.tensor([ids], device=next(network.parameters()).device)
    # Limit concurrent KV caches and avoid racing eval/train restoration.
    with _LOCK:
        output = network.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                  top_k=top_k, allowed_vocab_size=tokenizer.get_vocab_size())
    continuation = output[0, len(ids):].tolist()
    return {"generated_text": tokenizer.decode(continuation, skip_special_tokens=True),
            "generated_tokens": len(continuation)}
