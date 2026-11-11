"""Layout contracts that prevent divergent parent/worker/build import roots."""
from pathlib import Path
import unittest

import minimax_h3_amd_sp
import minimax_h3_attention
from minimax_h3_attention.paths import ROOT, SOURCE_ROOT, CACHE_ROOT


class LayoutTests(unittest.TestCase):
    def test_source_packages_share_one_source_root(self):
        for module in (minimax_h3_amd_sp, minimax_h3_attention):
            self.assertEqual(Path(module.__file__).resolve().parents[1], SOURCE_ROOT)

    def test_worker_script_is_inside_the_same_source_root(self):
        self.assertTrue((SOURCE_ROOT / "minimax_h3_amd_sp/worker.py").is_file())
        self.assertEqual(SOURCE_ROOT.parent, ROOT)

    def test_runtime_files_are_outside_public_source(self):
        self.assertEqual(CACHE_ROOT, ROOT / ".runtime/cache")
        self.assertTrue((ROOT / "configs/runtime-lock.json").is_file())

    def test_comfy_shim_has_no_model_implementation(self):
        shim = (ROOT / "nodes.py").read_text()
        self.assertIn("integrations.comfyui.nodes", shim)
        self.assertNotIn("class AMDMinimax", shim)
        self.assertTrue((ROOT / "integrations/comfyui/nodes.py").is_file())
