"""CPU contracts: no accidental backend changes or rank mismatch identities."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from minimax_h3_attention.config import AttentionConfig, BACKENDS
from minimax_h3_attention.dispatch import implementation


class ConfigurationTests(unittest.TestCase):
    def test_reduced_error_is_default(self):
        self.assertEqual(AttentionConfig().backend, "reduced_error")
        self.assertEqual(AttentionConfig().head_chunk, 7)
        self.assertEqual(AttentionConfig().query_tile, 128)

    def test_unknown_backend_is_rejected_without_fallback(self):
        with self.assertRaises(ValueError):
            AttentionConfig(backend="missing")

    def test_invalid_query_tile_and_head_chunk_are_rejected(self):
        for value in (0, 29, True, 1.5):
            with self.assertRaises(ValueError):
                AttentionConfig(head_chunk=value)
        for value in (256, True):
            with self.assertRaises(ValueError):
                AttentionConfig(query_tile=value)

    def test_br512_cannot_silently_select_other_backend(self):
        for backend in BACKENDS:
            if backend != "reduced_error":
                with self.assertRaises(ValueError):
                    AttentionConfig(backend=backend, query_tile=512)

    def test_environment_selection_is_explicit(self):
        with patch.dict("os.environ", {"H3_ATTENTION_BACKEND": "fp8_qk_fp8_pv",
                                       "H3_ATTENTION_HEAD_CHUNK": "7",
                                       "H3_ATTENTION_QUERY_TILE": "128"}, clear=True):
            self.assertEqual(AttentionConfig.from_environment().backend, "fp8_qk_fp8_pv")

    def test_rank_identity_detects_backend_or_tile_change(self):
        controls = [AttentionConfig(backend=b).digest() for b in BACKENDS]
        controls.append(AttentionConfig(query_tile=512).digest())
        self.assertEqual(len(controls), len(set(controls)))

    def test_package_import_does_not_import_torch_or_require_archive(self):
        code = "import minimax_h3_attention; import sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1] / "src")

    def test_each_native_control_resolves_to_its_own_entry(self):
        class Fake:
            attention = object()
            int8_qk_fp8_pv = object()
            fp8_qk_fp8_pv = object()
        with patch("minimax_h3_attention.dispatch.import_module", return_value=Fake) as loader:
            self.assertIs(implementation(AttentionConfig()), Fake.attention)
            self.assertEqual(loader.call_args.args[0], "minimax_h3_attention.ix.prefix_candidate")
            for path in ("int8_qk_fp8_pv", "fp8_qk_fp8_pv"):
                self.assertIs(implementation(AttentionConfig(backend=path)), getattr(Fake, path))


if __name__ == "__main__":
    unittest.main()
