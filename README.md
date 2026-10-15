# Tabula Nova 8B

A separate research successor to [Tabula-104M](../../tabula/README.md), targeting **8,023,977,984 trainable parameters**. It preserves Tabula's bias-free decoder, headwise QK-RMSNorm, grouped-query attention, SwiGLU, tied embeddings, digit-aware tokenizer and verified procedural tasks. It adds an 8K context configuration, compact KV caching, native CUDA GQA, BF16/FSDP2 training and a self-contained OCI export path.

Start with [the ordered workflow](docs/START_HERE.md) and [project status](PROJECT_STATUS.md). `tools/workflow.sh` runs collection, cleaning, mixing, preparation and tiny validation one stage at a time. [The data workspace](data/README.md) keeps source records and manifests separate from prepared training artifacts.

The workspace includes a small trained smoke checkpoint; **there are no trained 8B weights**. Better accuracy than Tabula requires controlled training and held-out evaluation. No full 8B training or cloud deployment has been performed.

| Setting | Tabula baseline | Nova 8B |
| --- | ---: | ---: |
| Parameters | 103,845,440 | 8,023,977,984 |
| Layers / width | 28 / 576 | 32 / 4,096 |
| Query / KV heads | 9 / 3 | 32 / 8 |
| SwiGLU width | 1,536 | 16,640 |
| Vocabulary capacity | 8,192 | 32,768 |
| Maximum context | 1,024 | 8,192 |
| Decoding | Recompute context | Compact per-layer KV cache |
| Training | Local GPU / TPU | CPU pilot; CUDA BF16 / FSDP2 |

The default uses inexpensive headwise attention gates. `8b-baseline` removes gates, and `8b-attnres` adds Tabula's experimental depth mixing. `pilot` retains approximately Tabula's dimensions; `tiny` is for correctness checks. Gates and depth mixing are hypotheses about quality, not established Nova results. RoPE uses theta 500,000; 8K must be learned through training, rather than inferred from a config change. Native GQA requests the PyTorch backend; kernel selection and speed remain hardware-dependent.

Install into a dedicated environment, using a CUDA PyTorch 2.8 build compatible with the OCI GPU image. `torch` is pinned to the 2.8 release series because FSDP2 and DCP APIs are version-sensitive.

```bash
cd research/tabula_nova_8b
python -m pip install -e '.[oci]'
python -m tabula_nova inspect --preset 8b --verify-meta
python -m unittest discover -s tests -v
```

Without dependencies, `PYTHONPATH=src python3 -m tabula_nova inspect` still reports exact parameter and memory estimates. `--verify-meta` requires PyTorch and checks the formula against an allocation-free model instantiation.

Start with a small end-to-end run:

```bash
python -m tabula_nova prepare --output prepared/smoke --vocab-size 512 \
  --target-tokens 50000 --tokenizer-sample-docs 128 --packed-seq-len 128
python -m tabula_nova train --preset tiny --device cpu --data prepared/smoke \
  --output runs/smoke --max-steps 3 --warmup-steps 0 --seq-len 64 \
  --accumulation-steps 2 --eval-every 1 --eval-batches 2 --save-every 1
python -m tabula_nova generate --checkpoint runs/smoke/latest.pt --device cpu \
  --prompt 'Compute 3 + 4.' --max-new-tokens 16
```

The tiny run verifies execution; its generated answers should not be expected to be useful. For a Tabula-sized pilot, use `--preset pilot`, vocabulary 8,192 and sequence length 512 or 1,024. Native tests skip when dependencies are absent; a successful dependency-free test run is not evidence that tensor execution works.

Prepare Tabula reasoning data with `prepare`. For general text/code or an externally assembled mixed corpus, use streaming JSONL with one `{"text": "..."}` record per document:

```bash
python -m tabula_nova prepare-text --input corpus.jsonl --output prepared/mixed \
  --vocab-size 32768
```

Text preparation fits the tokenizer only on content-hashed training documents, preserves identical text in the same split, and requires examples in all three splits. It reads the corpus twice without materializing it. Records should contain ordinary text, not literal special-token delimiters. It does not deduplicate, rebalance domains or apply Tabula's procedural input loss mask. The inherited procedural generator is finite-template data; increasing token count alone cannot establish general language ability. See [the research plan](docs/RESEARCH.md) for corpus and evaluation targets.

For OCI GPU training, export and model catalog registration, follow [OCI.md](docs/OCI.md). The job configuration starts at 2K training sequences for memory preflight. Extend to 4K and 8K in later training runs; training at 2K alone does not validate 8K quality. The 10,000-step file is a pilot schedule: at eight GPUs it processes at most 5.24B sequence positions, with fewer supervised tokens on padded/masked data.

```bash
python -m tabula_nova pack-job --output dist/tabula_nova_8b.zip
# In OCI's published GPU environment, with mounted data/checkpoint paths:
NOVA_DATA_DIR=/mnt/nova/prepared NOVA_OUTPUT_DIR=/mnt/nova/runs/8b \
  NOVA_GPUS_PER_NODE=8 bash oci/train.sh
```

Budget estimates: BF16 weights occupy **14.95 GiB**, a single 8K BF16 KV cache **1 GiB**, and FP32 parameters/gradients/AdamW moments **119.57 GiB** before sharding. FSDP2 at eight ranks divides persistent state to about **14.95 GiB per rank**, before activations, all-gathers, root weights and CUDA buffers. These are arithmetic estimates, not measured memory usage. A 6 GB laptop GPU cannot train this configuration from scratch. The code uses per-layer activation checkpointing and recomputes chunked vocabulary projections during backward.

Resume a local run with `--resume runs/smoke/latest.pt`, or an OCI run with `NOVA_RESUME=/mnt/nova/runs/8b/step-00000100`. Checkpoints require the same data hashes, architecture, world size and full training schedule. FSDP checkpoints are directories with DCP shards, per-rank RNG state and a completion metadata file; they are distinct from exported inference artifacts.

```bash
python -m tabula_nova benchmark --artifact exports/nova --suite template \
  --mode answer --context-limit 1024
python -m tabula_nova benchmark --artifact exports/nova --suite depth \
  --mode trace --context-limit 8192 --max-new-tokens 512
```

Use matched tokenizers, data, seeds, budgets and context limits when comparing with the original Tabula; report skipped overlong cases alongside exact-match scores. The exported safetensors index is a Nova artifact, not a Hugging Face Transformers/vLLM architecture registration. Use Nova's loader or OCI `score.py` for inference.

[VALIDATION.md](VALIDATION.md) records checks and remaining hardware validation. [PROVENANCE.md](PROVENANCE.md) identifies inherited source files.
