import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tabula_nova.workspace import clean, collect_web, generate_synthetic, mix, sha256, status


class WorkspaceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow") and importlib.util.find_spec("huggingface_hub"),
                         "data dependencies are not installed")
    def test_collector_pins_revision_bounds_sample_and_closes_local_reader(self):
        class Reader:
            closed = False
            def __init__(self, path):
                pass
            def iter_batches(self, **kwargs):
                if kwargs["use_threads"] is not False:
                    raise AssertionError("pilot reader must disable Arrow threads")
                yield SimpleNamespace(to_pylist=lambda: [{"text": f"Document {i}", "id": str(i), "url": None} for i in range(200)])
            def close(self):
                Reader.closed = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data").mkdir()
            source = {"id": "fineweb-edu", "dataset": "owner/dataset", "file_prefix": "sample/10BT/",
                      "text_field": "text", "dataset_license_label": "ODC-BY"}
            (root/"data/sources.json").write_text(json.dumps({"sources": [source]}))
            shard = root/"fixture.parquet"
            shard.write_bytes(b"mocked shard")
            api = SimpleNamespace(dataset_info=lambda name: SimpleNamespace(sha="pinned-revision"),
                                  list_repo_files=lambda *args, **kwargs: ["sample/10BT/a.parquet", "other/b.parquet"])
            with patch("huggingface_hub.HfApi", return_value=api), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(shard)) as fetch, \
                 patch("pyarrow.parquet.ParquetFile", Reader):
                report = collect_web(root, documents=10)
            self.assertEqual(report["documents"], 10)
            self.assertEqual(report["resolved_revision"], "pinned-revision")
            self.assertEqual(fetch.call_args.kwargs["revision"], "pinned-revision")
            self.assertTrue(Reader.closed)
            records = (root/"data/raw/web-v1/documents.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 10)
            self.assertEqual(len({json.loads(line)["source_id"] for line in records}), 10)

    def test_clean_deduplicates_across_inputs_preserves_provenance_and_reports_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root/"a.jsonl", root/"b.jsonl"
            text = "This is a useful document with enough text to pass our minimum character length. "*2
            a.write_text(json.dumps({"text": text, "source": "a", "url": "https://example.com"})+"\n")
            b.write_text(json.dumps({"text": text.replace(" ", "  "), "source": "b"})+"\n"+
                         json.dumps({"text": "short"})+"\n"+"{bad json\n")
            report = clean([a, b], root/"clean")
            self.assertEqual(report["counts"]["accepted"], 1)
            self.assertEqual(report["counts"]["rejected_duplicate"], 1)
            self.assertEqual(report["counts"]["rejected_length"], 1)
            self.assertEqual(report["counts"]["rejected_invalid_record"], 1)
            row = json.loads((root/"clean/documents.jsonl").read_text())
            self.assertEqual(row["source"], "a")
            self.assertEqual(row["url"], "https://example.com")
            self.assertEqual(report["sha256"]["documents.jsonl"], sha256(root/"clean/documents.jsonl"))
            with self.assertRaises(FileExistsError):
                clean([a], root/"clean")

    def test_shuffle_preserves_every_record_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root/"input.jsonl"
            records.write_text("".join(json.dumps({"text": f"Document {index}", "source": str(index%2)})+"\n" for index in range(100)))
            left, right = mix(records, root/"a", buffer_size=10), mix(records, root/"b", buffer_size=10)
            a, b = (root/"a/documents.jsonl").read_text(), (root/"b/documents.jsonl").read_text()
            self.assertEqual(a, b)
            self.assertNotEqual(a, records.read_text())
            self.assertEqual(set(a.splitlines()), set(records.read_text().splitlines()))
            self.assertEqual(left["documents"], 100)

    def test_failed_clean_does_not_publish_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/"empty.jsonl"
            raw.write_text('{"text":"tiny"}\n')
            with self.assertRaises(ValueError):
                clean([raw], root/"output")
            self.assertFalse((root/"output").exists())

    def test_synthetic_collection_is_reproducible_and_uses_train_families(self):
        from tabula_nova.synthetic import TRAIN_TOPICS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one", "two"):
                (root/name/"data/raw").mkdir(parents=True)
                generate_synthetic(root/name, documents=100, seed=42)
            a = (root/"one/data/raw/synthetic-v1/documents.jsonl").read_text()
            b = (root/"two/data/raw/synthetic-v1/documents.jsonl").read_text()
            self.assertEqual(a, b)
            for row in map(json.loads, a.splitlines()):
                self.assertIn(row["topic"], TRAIN_TOPICS[row["domain"]])
                self.assertEqual(row["source"], "tabula-synth-v3")


if __name__ == "__main__":
    unittest.main()
