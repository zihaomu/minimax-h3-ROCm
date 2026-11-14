import unittest
from unittest.mock import patch

from minimax_h3_amd_sp.cache_dit import CacheDiTConfig, make_cache


class CacheConfigTests(unittest.TestCase):
    def test_off_needs_no_optional_dependency(self):
        with patch.dict("os.environ", {}, clear=True), patch(
                "minimax_h3_amd_sp.cache_dit._upstream", side_effect=AssertionError):
            self.assertFalse(CacheDiTConfig.from_environment().enabled)
            self.assertIsNone(make_cache())

    def test_only_explicit_one_enables_cache(self):
        with patch.dict("os.environ", {"H3_CACHE_DIT": "1"}, clear=True):
            self.assertEqual(CacheDiTConfig.from_environment(), CacheDiTConfig(enabled=True))
        for flag in ("true", "auto", "2"):
            with patch.dict("os.environ", {"H3_CACHE_DIT": flag}, clear=True):
                with self.assertRaises(ValueError):
                    CacheDiTConfig.from_environment()

    def test_invalid_policy_is_rejected(self):
        for options in ({"fn": 0}, {"fn": 49, "bn": 1}, {"threshold": float("nan")},
                        {"threshold": 1}, {"threshold": -0.1}, {"warmup": 0}, {"max_continuous": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CacheDiTConfig(**options)
