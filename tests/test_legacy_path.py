"""Legacy queries must survive removing duplicate local research assets."""
from pathlib import Path
import tempfile
import unittest

from tools.legacy_path import resolve_legacy_path


class LegacyPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "archive/legacy-20261111/doc"
        self.fixture = "9700-standard/c1/inputs.safetensors"
        p = self.archive / self.fixture
        p.parent.mkdir(parents=True)
        p.write_bytes(b"synthetic fixture; no model data")

    def test_old_doc_query_falls_back_to_original_archive(self):
        self.assertEqual(resolve_legacy_path("doc/" + self.fixture, self.root),
                         self.archive / self.fixture)

    def test_explicit_vibe_history_query_uses_same_archive_tree(self):
        self.assertEqual(resolve_legacy_path("vibe/history/" + self.fixture, self.root),
                         self.archive / self.fixture)

    def test_absolute_old_doc_path_still_resolves(self):
        self.assertEqual(resolve_legacy_path(self.root / "doc" / self.fixture, self.root),
                         self.archive / self.fixture)

    def test_existing_local_document_takes_priority(self):
        p = self.root / "vibe" / self.fixture
        p.parent.mkdir(parents=True)
        p.write_bytes(b"local fixture")
        self.assertEqual(resolve_legacy_path("doc/" + self.fixture, self.root), p)

    def test_missing_file_is_not_claimed_to_exist(self):
        p = resolve_legacy_path("doc/9700-standard/missing.json", self.root)
        self.assertFalse(p.exists())
        self.assertEqual(p, self.root / "vibe/9700-standard/missing.json")

    def test_parent_traversal_and_outside_paths_are_rejected(self):
        for value in ("doc/../outside", self.root.parent / "outside", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_legacy_path(value, self.root)
