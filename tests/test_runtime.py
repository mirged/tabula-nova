"""Meaningful tensor/integration checks; skipped if dependencies are absent."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_EXPORT = HAS_TORCH and all(importlib.util.find_spec(name) is not None for name in ("tokenizers", "numpy", "safetensors", "tqdm"))


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self, **overrides):
        from tabula_nova.config import model_preset
        from tabula_nova.model import TabulaNova
        return TabulaNova(model_preset("tiny", **overrides))

    def test_exact_8b_count_on_meta(self):
        import torch
        from tabula_nova.config import model_preset
        from tabula_nova.model import TabulaNova
        with torch.device("meta"):
            for preset in ("8b", "8b-baseline", "8b-attnres"):
                config = model_preset(preset)
                self.assertEqual(TabulaNova(config).parameter_count, config.parameter_count)

    def test_full_cache_logits_match_for_chunks_and_single_tokens(self):
        import torch
        for depth in (0, 1):
            torch.manual_seed(3)
            model = self.model(residual_block_size=depth).eval()
            ids = torch.tensor([[1, 4, 5, 6, 7, 8]])
            with torch.no_grad():
                expected = model(ids)
                a, cache = model(ids[:, :2], use_cache=True)
                b, cache = model(ids[:, 2:5], use_cache=True, past_key_values=cache)
                c, cache = model(ids[:, 5:], use_cache=True, past_key_values=cache)
            torch.testing.assert_close(torch.cat((a, b, c), 1), expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(cache[0][0].shape, (1, 2, 6, 16))

    def test_causal_prefix_does_not_see_future(self):
        import torch
        model = self.model().eval()
        with torch.no_grad():
            first = model(torch.tensor([[1, 4, 5, 6]]))
            second = model(torch.tensor([[1, 4, 99, 100]]))
        torch.testing.assert_close(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-5)

    def test_checkpointed_chunk_loss_and_gradients_match_full_projection(self):
        import torch
        a, b = self.model(loss_chunk_size=2), self.model(loss_chunk_size=0, gradient_checkpointing=False)
        b.load_state_dict(a.state_dict())
        ids = torch.tensor([[1, 4, 5, 6, 7]])
        targets = torch.tensor([[4, -100, 6, 7, 0]])
        left, right = a(ids, targets), b(ids, targets)
        torch.testing.assert_close(left, right)
        left.backward()
        right.backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)

    def test_all_ignored_loss_has_finite_zero_gradient(self):
        import torch
        model = self.model()
        ids = torch.tensor([[1, 4, 5]])
        loss = model(ids, torch.full_like(ids, -100))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cached_generation_matches_reference_and_restores_mode(self):
        import torch
        model = self.model()
        ids = torch.tensor([[1, 4, 5]])
        a = model.generate(ids, 8, temperature=0)
        b = model.generate(ids, 8, temperature=0, use_cache=False)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            model.generate(torch.ones((1, 129), dtype=torch.long))

    @unittest.skipUnless(HAS_EXPORT, "full runtime dependencies are not installed")
    def test_native_cuda_gqa_matches_explicit_reference(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA runtime unavailable (CPU wheel or no GPU)")
        a, b = self.model(native_gqa=True).cuda().eval(), self.model(native_gqa=False).cuda().eval()
        b.load_state_dict(a.state_dict())
        ids = torch.tensor([[1, 4, 5, 6]], device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(a(ids), b(ids), atol=1e-5, rtol=1e-4)


@unittest.skipUnless(HAS_EXPORT, "PyTorch/tokenizers/numpy/safetensors/tqdm are not installed")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_prepare_train_resume_export_load_and_score(self):
        import torch
        from tabula_nova.config import model_preset
        from tabula_nova.corpus import prepare_text
        from tabula_nova.export import export_checkpoint
        from tabula_nova.inference import load_artifact
        from tabula_nova.training import train, TrainConfig, load_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root/"corpus.jsonl"
            corpus.write_text("\n".join(json.dumps({"text": f"Document {i}: compute {i} plus 7. Answer: {i+7}."}) for i in range(1000)))
            prepare_text([corpus], root/"data", vocab_size=512, sample_documents=64)
            config = TrainConfig(seq_len=16, accumulation_steps=2, max_steps=3, warmup_steps=0,
                                 eval_every=1, eval_batches=2, save_every=1)
            model_config = model_preset("tiny")
            train(root/"data", root/"reference", model_config, config, "cpu")
            # Stop immediately after the first committed checkpoint, without
            # modifying the saved full schedule or optimizer state.
            from tabula_nova import training
            original_save = training.atomic_save
            def stop_after_save(value, path):
                original_save(value, path)
                raise InterruptedError("test interruption after saved update")
            with patch("tabula_nova.training.atomic_save", side_effect=stop_after_save):
                with self.assertRaises(InterruptedError):
                    train(root/"data", root/"resumed", model_config, config, "cpu")
            train(root/"data", root/"resumed", model_config, config, "cpu", resume=root/"resumed/latest.pt")
            reference = load_checkpoint(root/"reference/latest.pt")
            resumed = load_checkpoint(root/"resumed/latest.pt")
            self.assertEqual(reference["tokens_seen"], resumed["tokens_seen"])
            for key in reference["model"]:
                torch.testing.assert_close(reference["model"][key], resumed["model"][key], atol=0, rtol=0)
            artifact = root/"artifact"
            export_checkpoint(root/"resumed/latest.pt", artifact, "gpu_test", "oci://bucket@namespace/env.tar.gz", shard_size_mib=1)
            network, tokenizer = load_artifact(artifact, "cpu")
            self.assertEqual(network.parameter_count, model_config.parameter_count)
            import importlib.util
            specification = importlib.util.spec_from_file_location("nova_score", artifact/"score.py")
            score = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(score)
            result = score.predict({"prompt": "Compute 3 plus 4.", "max_new_tokens": 4}, (network, tokenizer))
            self.assertIn("generated_text", result)
            with (artifact/"config.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(artifact, "cpu")

    def test_two_gpu_fsdp_checkpoint_resume_and_partial_export(self):
        import torch
        if torch.cuda.device_count() < 2 or not torch.cuda.is_bf16_supported():
            self.skipTest("two BF16-capable CUDA GPUs are required")
        from tabula_nova.corpus import prepare_text
        from tabula_nova.export import export_checkpoint
        from tabula_nova.inference import load_artifact
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root/"corpus.jsonl"
            corpus.write_text("\n".join(json.dumps({"text": f"Document {i}: calculate {i} plus 5."}) for i in range(1000)))
            prepare_text([corpus], root/"data", vocab_size=512, sample_documents=64)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                       "-m", "tabula_nova", "train", "--preset", "tiny", "--device", "cuda", "--fsdp",
                       "--data", str(root/"data"), "--output", str(root/"runs"), "--max-steps", "2",
                       "--warmup-steps", "0", "--seq-len", "16", "--accumulation-steps", "2",
                       "--eval-every", "1", "--eval-batches", "2", "--save-every", "1"]
            subprocess.run(command, check=True, timeout=180)
            # Resume from update one with the saved full two-step schedule.
            command[command.index(str(root/"runs"))] = str(root/"resumed")
            subprocess.run(command+["--resume", str(root/"runs/step-00000001")], check=True, timeout=180)
            for source, target in ((root/"runs/step-00000002", root/"artifact-a"),
                                   (root/"resumed/step-00000002", root/"artifact-b")):
                export_checkpoint(source, target, "test_gpu", "oci://bucket@namespace/env.tar.gz", shard_size_mib=1)
            a, _ = load_artifact(root/"artifact-a", "cpu")
            b, _ = load_artifact(root/"artifact-b", "cpu")
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
