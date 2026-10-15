# Validation status

Checked locally on 2026-10-15. The initial restricted session could only run dependency-free tests. Network access is now available, and a dedicated project `.venv` has been installed with Python 3.13, CPU PyTorch 2.8, NumPy, tokenizers, safetensors, dataset tools and OCI SDK. `reports/environment.json` and `environment-lock.txt` record the current environment. The host has an RTX 4050 with 6 GB VRAM; the installed CPU wheel does not expose CUDA.

## Checks completed

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash -n oci/train.sh
python3 -m compileall -q src tests oci
PYTHONPATH=src python3 -m tabula_nova inspect
PYTHONPATH=src python3 -m tabula_nova pack-job --output dist/tabula_nova_8b.zip
```

The first dependency-free run passed 15 checks and skipped nine runtime tests. The current native CPU run is recorded in `reports/native-tests.json`; only CUDA and two-GPU checks require a different runtime. Checks cover architecture constraints, memory/cache arithmetic, sampling/resume offsets, Python syntax, Oracle runtime schema, shard planning, archive layouts, transfer helpers, cleaning/deduplication and seeded mixing. They also confirm the four inherited data/benchmark/progress files are byte-identical to Tabula.

The architecture formula and actual allocation-free PyTorch meta-model both confirm **8,023,977,984 parameters**. Baseline and AttnRes preset counts also pass. `reports/architecture.json` records meta verification. The source job ZIP contains the matching project root, training entrypoint, configs, workflow tools and source registry; it contains no weights, dataset or environment credentials.

Native CPU tests pass causal-prefix checks, full-vs-cached logits with single-token and multi-token appends, ordinary/AttnRes cache behavior, cached-vs-reference generation, checkpointed chunk-loss gradient equivalence and finite all-masked gradients. Integration prepares native text data, trains a tiny model, interrupts/resumes, compares final weights exactly, exports/reloads BF16 weights, calls the scoring function locally and detects checksum corruption. The two-GPU FSDP2/DCP and CUDA GQA tests remain skipped.

The organized workspace contains 1,000 FineWeb-Edu documents and 1,000 assertion-checked synthetic examples. Cleaning accepted 1,998 records and rejected two on length, then deterministic mixing and smoke/pilot tokenization completed. A saved three-update tiny CPU run trained on 384 supervised tokens and produced finite training/validation losses and `runs/workflow-smoke/latest.pt`. These results demonstrate pipeline execution, not meaningful task quality.

Remote Parquet streaming initially aborted at interpreter shutdown after publishing complete, checksum-valid files. Explicit closure and disabling pre-buffering did not reliably resolve the upstream Arrow shutdown problem. The collector now caches needed shards and reads local batches with `use_threads=False`. Source/revision/shard checksums are preserved; collection logs record the actual local-reader validation. [Related upstream report](https://github.com/huggingface/datasets/issues/8793).

## Required before a substantial run

Run CUDA/two-GPU checks in their target environments. Before scaling the data, add near-duplicate detection, grouped splits, sensitive-data filtering and benchmark contamination checks. On OCI, run the three-update GPU preflight, measure allocated memory and throughput, verify finite gradients and restart a committed DCP checkpoint. Validate the partial-DCP exporter and the runtime's published Conda environment.

OCI SDK transfers/catalog registration have been checked against official documentation and mocked locally; actual SDK requests, IAM policies, catalog work requests and Model Deployment scoring are untested. No OCI account connection, uploaded model or deployment was created.

Only tiny smoke weights have been trained. There are no trained 8B Nova weights, measured task quality, accuracy advantage, measured inference speedup or demonstrated 8K-context quality. Establish those with the protocol in [RESEARCH.md](docs/RESEARCH.md).
