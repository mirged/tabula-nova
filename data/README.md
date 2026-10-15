# Data workspace

`sources.json` defines the initial English-language pilot sources. Raw collection records include source metadata and a resolved Hugging Face revision. Every stage publishes a new directory only after completion and refuses to replace an existing version.

| Folder | Contents |
| --- | --- |
| `raw/web-v1/` | Bounded FineWeb-Edu sample and collection manifest |
| `raw/synthetic-v1/` | Assertion-checked Tabula examples and generator manifest |
| `cleaned/pilot-v1/` | Accepted documents, rejected record references, SQLite dedup index, report |
| `mixed/pilot-v1/` | Seeded shuffled documents and input/output hashes |
| `cache/` | Regenerable Hugging Face cache |
| `manifests/` | Optional cross-run corpus inventories |

Large corpus files and caches are ignored by Git. Keep source definitions, manifests and reports with experiment records. Back up collected/cleaned data before removing caches or transferring work to OCI.

The first sample is a pipeline check. Equal requested document counts do not imply equal token counts. The mixer preserves records and provenance but does not enforce the proposed final token proportions. Cleaning provides exact normalized deduplication and simple format/length/repetition filters; near-duplicate detection, grouped splits, sensitive-data filtering and contamination audits remain required work for a production corpus.

All raw/cleaned/mixed records are UTF-8 JSONL. Prepare them with a representative training-only tokenizer sample. The text preparation path trains on all non-padding tokens, including synthetic setup literals; use the inherited procedural `prepare` path separately when comparing Tabula's masked training recipe.
