# Research decisions and measurable improvement

Reviewed 2026-10-15 against primary research and official framework documentation. Nova is a code/architecture successor with untrained weights; claims below are design motivations or evaluation criteria.

## Architecture

The decoder retains Tabula's RMSNorm, SwiGLU, rotary positions, GQA and QK normalization. LLaMA provides evidence for this general decoder recipe; GQA studies the quality/decoding-memory tradeoff of sharing KV heads. Nova uses 32 query and 8 KV heads, storing only the eight KV heads in its cache. This reduces the cache size by 4x relative to a hypothetical 32-KV-head model of the same dimensions, without claiming a measured latency advantage. [LLaMA](https://arxiv.org/abs/2302.13971), [GQA](https://arxiv.org/abs/2305.13245).

Headwise sigmoid attention gates are retained as the default experiment. They add 4,194,304 parameters at 8B scale. The published gated-attention results motivate testing this choice, but do not establish an advantage for Nova's task mix. `8b-baseline` gives a controlled ungated comparison. Block Attention Residuals remain an opt-in configuration adding 532,480 parameters to the gated model. This implementation inherits Tabula's adaptation rather than reproducing a published distributed training stack. [Gated Attention](https://arxiv.org/abs/2505.06708), [Attention Residuals](https://arxiv.org/abs/2603.15031).

RoPE theta 500,000 and 8,192 maximum positions are configuration choices. Train on progressive 2K, 4K and 8K contexts, including long traces and variable binding. Run position/length tests before declaring effective 8K context. Cached decoding uses a lower-right causal mask for multi-token cache appends; a single appended token can see every preceding cache position. Full and cached logits must agree in runtime tests. Generation stops at the context boundary instead of silently dropping earlier variables.

PyTorch 2.8 SDPA supplies CUDA-native GQA; CPU retains the explicit-repeat reference. This requests a supported attention implementation and does not guarantee a particular fused kernel. The training package applies FSDP2 layer by layer before the root, with meta initialization, BF16 compute, FP32 reductions, activation checkpointing, chunked projection recomputation and AdamW. [SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html), [FSDP2](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html), [DCP state APIs](https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html).

## Data and training budget

Use a planning target of **160B tokens**, approximately 20 tokens per parameter, then determine the real budget from pilot loss curves, diversity and hardware throughput. The Chinchilla paper motivates balancing data and model size; it does not establish that exactly 160B tokens are optimal for Tabula-derived procedural learning. The approximate `6NT` compute estimate is 7.70e21 FLOPs, excluding attention and overhead. [Compute-optimal training study](https://arxiv.org/abs/2203.15556).

A starting hypothesis for externally assembled token-balanced data is 50% general text, 20% code and 30% verified procedural reasoning. Within procedural examples, retain Tabula's 30/25/25/20 algorithm/logic/math/grammar split. The code supports procedural preparation or streaming JSONL text input; a combined tokenizer, deduplication, cross-source contamination audit and adaptive domain weighting are still corpus engineering work. Do not independently tokenize corpora and concatenate their binary IDs. Prepare the assembled mixture with one frozen tokenizer.

The synthetic corpus gives repeatable assertion-checked labels and held-out topic/depth/length suites. It remains limited by template diversity and conservatively masks numeric setup literals. Fresh random values cannot substitute for general text and code. A 1M-token preparation default is a pipeline pilot, not an 8B pretraining budget. A 10,000-step eight-rank schedule at 2K context contains at most 5.24B positions; padding and masked labels reduce supervised tokens. Use reported `tokens_seen` for the actual trained target count.

## Establishing better performance

1. Run tiny execution checks and a Tabula-sized pilot first, including native tokenizer audits and checkpoint restart.
2. Compare Tabula baseline and Nova `pilot` on identical prepared data/tokenizer, seeds, supervised token budgets, context limits, and answer/trace metrics. This isolates the code-level gate experiment near the original scale.
3. Train Nova `8b-baseline`, `8b` and optionally `8b-attnres` on a shared fixed corpus and schedule, reporting parameter/token/compute budgets explicitly. Repeat promising differences across seeds.
4. Record per-domain held-out loss, answer exact match, trace-step accuracy, template/depth/length generalization, overlong-case counts and token efficiency. Benchmark prompts withhold solved traces. Use the same test cases and context limits for comparisons; larger-context results should be reported separately.
5. For general language capability, add externally maintained code/text benchmarks with contamination checks. The inherited procedural benchmark alone cannot establish a generally better LLM.
6. Measure GPU peak memory, steady throughput, cache-vs-recompute latency, checkpoint I/O, OCI artifact verification and endpoint output equivalence. Run FSDP/DCP and deployment tests on the actual OCI GPU image before a long run.

No quality score, speedup, cost or training ETA is asserted in this project. The original Tabula checkpoint is not directly transferable to the much wider/deeper Nova architecture; training begins from scratch.
