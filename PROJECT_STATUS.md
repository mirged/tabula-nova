# Project status

Updated 2026-10-15. Follow [START_HERE.md](docs/START_HERE.md) for the ordered workflow. The current objective is an English-language, Tabula-derived 8B model with a reproducible dataset and OCI handoff.

| Stage | Status | Evidence / next action |
| --- | --- | --- |
| Project layout and stage commands | Complete | `tools/workflow.sh`, source registry and versioned data folders |
| Isolated CPU environment | Complete | `reports/environment.json`, `environment-lock.txt` |
| Native CPU correctness | Complete | 27 tests pass; two GPU tests skipped in CPU runtime |
| 8B architecture count | Complete | Actual meta model confirms 8,023,977,984 parameters |
| First source samples | Complete | 1,000 FineWeb-Edu + 1,000 verified synthetic records |
| Cleaning and mixing | Complete | 1,998 accepted; two rejected on length; hashed manifests |
| Tiny training pipeline | Complete | Three CPU updates, finite losses, saved checkpoint |
| Tabula-sized pilot preparation | Complete | 1,554,588 training tokens; 8,192-token vocabulary |
| GPU pilot | Next | Install CUDA wheel, run CUDA checks, then bounded pilot schedule |
| Production data processing | Pending | Deduplicate near matches, filter sensitive data, audit contamination, group related sources, freeze/shard tokenizer |
| Larger corpus and mixture | Pending | Choose token ratios and add code sources; scale from measured pilot results |
| OCI GPU preflight | Pending | Select tenancy/project/region/topology/bucket/runtime/budget; run three-update FSDP2 preflight |
| Full 8B training | Pending | Requires a frozen substantial corpus and measured cloud throughput |
| 8B quality evaluation and deployment | Pending | Train, compare held-out results, export a candidate and verify OCI scoring |

The completed samples are pipeline data, not a sufficient 8B pretraining corpus. The smoke run trained only 384 supervised tokens; its checkpoint demonstrates execution rather than useful answers. No 8B weights or OCI deployment have been created.

The original `tabula/` project remains separate. This workspace uses a CPU PyTorch wheel for initial validation; the host's RTX 4050 is available to a CUDA-enabled environment. Do not interpret CPU-runtime GPU test skips as a claim that the machine has no GPU.

Inspect progress at any time:

```bash
bash tools/workflow.sh status
```

Collection shutdown diagnostics are preserved in reports. Published original sample files were checked against their hashes and counts. The collector has been changed to a cached local Parquet reader; use its source/shard manifest for repeatable future collection.
