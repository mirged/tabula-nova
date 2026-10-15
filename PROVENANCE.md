# Source provenance

Created from the local Tabula source tree on 2026-10-15. Original project: `../../tabula`. No original files were changed.

`src/tabula_nova/data.py`, `synthetic.py`, `benchmark.py`, and `progress.py` were copied unchanged to keep the procedural taxonomy, assertions, content-hash splits, masked uint16 data format and benchmark scoring comparable. `model.py` derives from Tabula's model implementation, extending it with separate scalable configuration, compact cached inference, native CUDA GQA and meta/FSDP initialization. Configuration, training, streaming text preparation, artifact loading/export and OCI helpers are new implementations.

This derivation reuses architecture and code; it does not upscale a trained Tabula checkpoint. Tabula-104M weights are dimensionally incompatible with Nova 8B and are rejected by Nova's checkpoint loader. Export records the training step and token count and contains inference weights only.
