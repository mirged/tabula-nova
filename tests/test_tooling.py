import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from tabula_nova.config import ModelConfig, model_preset, budget
from tabula_nova.export import runtime_config, shard_groups, zip_artifact
from tabula_nova.oci_job import pack_job
from tabula_nova.oci_storage import parse_uri, upload, download
from tabula_nova.sampling import StepSampler

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_8b_target_and_checked_in_configuration(self):
        config = model_preset()
        self.assertEqual(config.parameter_count, 8_023_977_984)
        self.assertLess(abs(config.parameter_count/8_000_000_000-1), 0.01)
        self.assertEqual(config, ModelConfig.load(ROOT/"configs/8b.json"))
        self.assertEqual(config.parameter_count-model_preset("8b-baseline").parameter_count, 4_194_304)
        self.assertEqual(model_preset("8b-attnres").parameter_count-config.parameter_count, 532_480)

    def test_budget_cache_and_sharding(self):
        small, large = budget(model_preset(), world_size=8), budget(model_preset(), world_size=1)
        self.assertEqual(small["kv_cache_gib"], 1)
        self.assertEqual(small["sharded_persistent_state_gib_per_rank"]*8,
                         large["sharded_persistent_state_gib_per_rank"])
        self.assertEqual(budget(replace(model_preset(), kv_heads=32))["kv_cache_gib"], 4)
        with self.assertRaises(ValueError):
            budget(model_preset(), context=8193)

    def test_invalid_architectures_fail_before_allocation(self):
        for values in ({"heads": 7}, {"kv_heads": 3}, {"dim": 0}, {"dim": True},
                       {"rope_theta": float("nan")}, {"eos_id": 1}, {"loss_chunk_size": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(**values)

    def test_all_python_sources_parse(self):
        paths = [path for directory in ("src", "tests", "tools", "oci")
                 for path in (ROOT/directory).rglob("*.py")]
        for path in paths:
            if "__pycache__" not in path.parts:
                with self.subTest(path=str(path)):
                    ast.parse(path.read_text(), filename=str(path))

    def test_derived_data_and_benchmark_are_identical(self):
        original = ROOT.parents[1]/"tabula/src/tabula"
        if not original.exists():
            self.skipTest("original Tabula checkout is not present")
        for name in ("data.py", "synthetic.py", "benchmark.py", "progress.py"):
            self.assertEqual((original/name).read_bytes(), (ROOT/"src/tabula_nova"/name).read_bytes())


class SamplingTests(unittest.TestCase):
    def config(self, **overrides):
        return SimpleNamespace(micro_batch_size=2, accumulation_steps=3, max_steps=5, seed=42, **overrides)

    def test_epoch_partition_and_resume(self):
        config = self.config()
        ranks = [list(StepSampler(60, config, rank, 2)) for rank in range(2)]
        # 60 samples total: each row is visited once across both ranks.
        self.assertEqual(len(ranks[0]), 30)
        self.assertEqual(set(ranks[0]).intersection(ranks[1]), set())
        self.assertEqual(set(ranks[0]+ranks[1]), set(range(60)))
        for rank in range(2):
            resumed = list(StepSampler(60, config, rank, 2, start_step=2))
            self.assertEqual(resumed, ranks[rank][12:])

    def test_composite_size_crosses_epoch_deterministically(self):
        config = self.config()
        rows = list(StepSampler(12, config))
        self.assertEqual(set(rows[:12]), set(range(12)))
        self.assertEqual(set(rows[12:24]), set(range(12)))
        self.assertEqual(rows, list(StepSampler(12, config)))
        with self.assertRaises(ValueError):
            StepSampler(1, config, world_size=2)


class ExportToolingTests(unittest.TestCase):
    def test_runtime_matches_oracle_schema_and_quotes_values(self):
        runtime = runtime_config("gpu:prod", "oci://bucket@namespace/conda/gpu.tar.gz", "3.11")
        parsed = json.loads(json.dumps(runtime))
        self.assertEqual(parsed["MODEL_ARTIFACT_VERSION"], "3.0")
        env = parsed["MODEL_DEPLOYMENT"]["INFERENCE_CONDA_ENV"]
        self.assertEqual(set(env), {"INFERENCE_ENV_SLUG", "INFERENCE_ENV_TYPE", "INFERENCE_ENV_PATH", "INFERENCE_PYTHON_VERSION"})
        self.assertEqual(env["INFERENCE_ENV_TYPE"], "published")
        with self.assertRaises(ValueError):
            runtime_config("gpu", "https://example.com/env")

    def test_shard_budget_never_splits_or_drops_a_tensor(self):
        shapes = {"embedding": (10, 8), "q": (8, 8), "norm": (8,)}
        groups = shard_groups(shapes, 180)
        self.assertEqual(groups, [["embedding"], ["q", "norm"]])
        self.assertEqual([key for group in groups for key in group], list(shapes))
        with self.assertRaises(ValueError):
            shard_groups(shapes, 150)

    def test_model_zip_has_root_scoring_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"artifact"
            root.mkdir()
            (root/"score.py").write_text("x = 1")
            (root/"runtime.yaml").write_text("{}")
            output = zip_artifact(root, Path(directory)/"model.zip")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {"score.py", "runtime.yaml"})
            with self.assertRaises(FileExistsError):
                zip_artifact(root, output)
            with self.assertRaises(ValueError):
                zip_artifact(root, root/"bad.zip")

    def test_job_zip_layout_and_dataset_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pack_job(Path(directory)/"tabula_nova_8b.zip")
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertIn("tabula_nova_8b/oci/train.sh", names)
                self.assertIn("tabula_nova_8b/src/tabula_nova/templates/score.py", names)
                self.assertTrue(all(name.startswith("tabula_nova_8b/") for name in names))
                self.assertFalse(any("prepared/" in name or name.endswith(".pt") for name in names))
            with self.assertRaises(ValueError):
                pack_job(Path(directory)/"wrong.zip")

    def test_inspect_and_pack_cli_without_runtime(self):
        result = subprocess.run([sys.executable, "-m", "tabula_nova", "inspect"], check=True,
                                capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["parameters"], 8_023_977_984)


class StorageTests(unittest.TestCase):
    def test_uri_validation(self):
        self.assertEqual(parse_uri("oci://bucket@namespace/nova/weights/"), ("bucket", "namespace", "nova/weights/"))
        for uri in ("https://bucket", "oci://bucket/path", "oci://@ns/path", "oci://b@n/", "oci://b@n/../outside", "oci://b@n/path?token=secret"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_uri(uri)

    def test_upload_preserves_directory_keys_using_multipart_manager(self):
        sdk = SimpleNamespace(object_storage=SimpleNamespace(UploadManager=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"part").mkdir()
            (root/"part/file").write_bytes(b"abc")
            with patch.dict(sys.modules, {"oci": sdk}), patch("tabula_nova.oci_storage.client"):
                result = upload(root, "oci://bucket@ns/models/nova/")
            self.assertEqual(result["files"], 1)
            sdk.object_storage.UploadManager.return_value.upload_file.assert_called_once_with(
                "ns", "bucket", "models/nova/part/file", str(root/"part/file"))

    def test_download_rejects_object_traversal(self):
        objects = [SimpleNamespace(name="prefix/../../outside")]
        sdk = SimpleNamespace(pagination=SimpleNamespace(list_call_get_all_results=Mock(
            return_value=SimpleNamespace(data=SimpleNamespace(objects=objects)))))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"oci": sdk}), patch("tabula_nova.oci_storage.client"):
                with self.assertRaisesRegex(ValueError, "unsafe object"):
                    download("oci://bucket@ns/prefix/", Path(directory)/"safe")


if __name__ == "__main__":
    unittest.main()
