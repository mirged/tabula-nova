# Follow these stages in order

Work from `research/tabula_nova_8b`. The first milestone is a reproducible small data pipeline and a tiny training checkpoint. The 8B architecture remains the final target. Initial language is English; add other languages through an explicit source/mix decision.

```bash
cd /home/mirged/Projects/active/ai/research/tabula_nova_8b
bash tools/workflow.sh status
```

Status shows which stage artifacts exist. Read `reports/native-tests.json` for actual test success and skips. Collection, cleaning, mixing and preparation refuse to overwrite existing versions. To extend a corpus, create a new version with explicit inputs/outputs rather than deleting the completed pilot.

The initial stages may already be completed in this workspace; inspect `PROJECT_STATUS.md` first and run the next pending stage. To collect a new version, use `collect --documents 10000 --output data/raw/web-v2`; synthetic generation also accepts `--output`. Pass the new files explicitly to `clean --input ... --output ...` and `mix --input ... --output ...`.

## 1. Install and inspect the environment

```bash
bash tools/workflow.sh setup cpu
bash tools/workflow.sh doctor
bash tools/workflow.sh test
```

The dedicated `.venv` contains PyTorch 2.8, dataset tools and OCI SDK. `reports/environment-lock.txt` records installed versions; `reports/environment.json` records hardware/runtime availability. CPU setup is enough for data work and tiny validation. On this RTX 4050 laptop, choose `setup cuda` when moving to the GPU pilot; it installs the separate CUDA 12.8 wheel. It does not change system Python or GPU drivers. [Official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/), [pip bootstrap](https://pip.pypa.io/en/stable/installation/).

Proceed when non-hardware tests pass and meta verification confirms the 8B parameter count. CUDA and two-GPU tests can remain skipped on CPU setup; run them in the corresponding GPU environment.

## 2. Collect a bounded text sample

```bash
bash tools/workflow.sh collect --documents 1000
```

The source registry is `data/sources.json`. This collects a FineWeb-Edu sample into `data/raw/web-v1/`, recording the dataset revision, source IDs, URLs and a SHA-256 manifest. It downloads only the needed Parquet shards into the local cache, then reads bounded batches with threading disabled. A whole shard is larger than the final 1,000-document sample; downloaded filenames, byte counts and hashes are recorded. Review source conditions and a few sample documents. [Dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

The local-shard reader avoids an observed crash when partially consumed remote Arrow streams shut down. Explicit iterator closure did not reliably prevent it. [Related upstream issue](https://github.com/huggingface/datasets/issues/8793).

## 3. Generate verified reasoning examples

```bash
bash tools/workflow.sh synthetic --documents 1000
```

`data/raw/synthetic-v1/` records the generator seed, version and domain counts. Use the `train` topic profile so held-out families stay excluded. Equal web/synthetic document counts are convenient for smoke validation; they are not the intended final token mixture.

## 4. Clean and deduplicate the combined sample

```bash
bash tools/workflow.sh clean
```

Read `data/cleaned/pilot-v1/manifest.json` and `rejected.jsonl`. Cleaning normalizes Unicode/line endings, removes malformed or out-of-range documents, rejects extreme line repetition and removes exact normalized duplicates across both sources using an on-disk SQLite index. It preserves source metadata.

This is a pilot filter. Near-duplicate detection, benchmark contamination checks, source/repository-grouped splits and sensitive-data filtering remain open production-corpus tasks. Do these before an expensive pretraining run. The current preparer uses 98/1/1 content-hash splits.

## 5. Shuffle before fitting the tokenizer

```bash
bash tools/workflow.sh mix
```

`data/mixed/pilot-v1/` contains a seeded buffer-shuffled copy with input/output hashes. This avoids fitting the tokenizer only on the first source. Mixing preserves every accepted document and uses bounded RAM; it does not enforce token ratios. For a production mixture, measure tokens per source with a shared tokenizer and explicitly resample by token budget.

## 6. Prepare and run a tiny smoke experiment

```bash
bash tools/workflow.sh prepare
bash tools/workflow.sh smoke
```

The smoke corpus uses a 512-token vocabulary under `prepared/workflow-smoke/`. Its three-update CPU run writes `runs/workflow-smoke/latest.pt` and `metrics.jsonl`. This verifies data/training/checkpoints, not useful learned accuracy. Resume with:

```bash
bash tools/workflow.sh smoke --resume runs/workflow-smoke/latest.pt
```

The native integration test also verifies interrupted-run equivalence and inference export/reload. It is separate from this saved demonstration run. The mixed JSONL path supervises all tokens; it does not preserve the original procedural setup loss mask.

## 7. Start a Tabula-sized GPU pilot

```bash
bash tools/workflow.sh setup cuda
bash tools/workflow.sh test
bash tools/workflow.sh prepare-pilot
.venv/bin/python -m tabula_nova train --preset pilot --device cuda \
  --data prepared/workflow-pilot --output runs/workflow-pilot \
  --train-config configs/train-pilot.json
```

The configuration starts at 512-token sequences, microbatch one and 100 updates. Measure memory and throughput before a longer run. The small collected sample may be repeated and is only a pilot. Compare the original Tabula and Nova on controlled data and report held-out exact-match accuracy, not only training loss. Keep smoke and pilot tokenizers separate; their binary IDs are incompatible.

## 8. Expand data and freeze a corpus version

Choose target language coverage and coding/reasoning priorities. Add code only after arranging source access and defining license/secret/repository handling. Keep instruction-tuning examples as a later stage. Record each dataset revision and acquisition manifest. Expand template diversity instead of filling the corpus with repeated variations of the same tasks.

Create a final mixed corpus, fit a representative training-only 32,768-capacity tokenizer, then freeze it for all corpus shards. The existing preparer runs over local JSONL inputs twice and is not a resumable distributed 160B-token preprocessing system. Implement frozen-tokenizer reuse/sharded preparation and production filtering before that scale. The 160B-token estimate is a planning budget, not a minimum to download now.

## 9. Move a measured preflight to OCI

Follow [OCI.md](OCI.md) after selecting an OCI region, GPU topology, compartment/project, bucket, persistent filesystem, runtime environment and budget. Record those choices in a private local deployment configuration. Account IDs and environment choices cannot be inferred from the source tree.

Prepare the source job archive and use `configs/train-oci-preflight.json` for a three-update run. Use a separate output path. Verify multi-GPU FSDP2, checkpoint restart and shard export before launching the real schedule. The full schedule remains `configs/train-oci.json`. A long 8B run needs much more data than this sample.

## 10. Evaluate and export a trained candidate

Run held-out answer/trace/topic/depth/length suites; compare matched contexts and budgets. Export the chosen trained checkpoint with `export-oci`, upload the ZIP through Object Storage, register it, then test the scoring endpoint. Keep training metrics, dataset hashes and export manifests with the experiment record in `experiments/`.

Use `experiments/run-template.json` to record a run before launching it. Update `PROJECT_STATUS.md` when a milestone changes; generated reports provide the underlying evidence.
