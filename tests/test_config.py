import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import ConfigError, config_from_mapping


class ConfigTests(unittest.TestCase):
    def test_relative_paths_resolve_from_config_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config = config_from_mapping(
                {
                    "input_sources": [{"path": "inputs", "type": "pdf"}],
                    "output_root": "work/output",
                    "log_dir": "work/logs",
                    "cache_root": "work/cache",
                },
                base_dir=base,
            )
            self.assertEqual(config.input_sources[0].path, (base / "inputs").resolve())
            self.assertEqual(config.manifest_path, (base / "work/logs/manifest.sqlite3").resolve())
            self.assertEqual(config.runtime.ocr_workers, 1)
            self.assertFalse(config.overwrite)
            self.assertTrue(config.resume)

    def test_rejects_non_positive_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "pdf"}],
                        "output_root": "work/output",
                        "log_dir": "work/logs",
                        "cache_root": "work/cache",
                        "runtime": {"ocr_workers": 0},
                    },
                    base_dir=base,
                )

    def test_rejects_string_boolean_instead_of_coercing_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "pdf"}],
                        "output_root": "output",
                        "log_dir": "logs",
                        "cache_root": "cache",
                        "overwrite": "false",
                    },
                    base_dir=base,
                )

    def test_rejects_malformed_gpu_selector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "image"}],
                        "output_root": "out",
                        "log_dir": "logs",
                        "cache_root": "cache",
                        "runtime": {"device": "gpu:"},
                    },
                    base_dir=base,
                )

    def test_rejects_output_nested_inside_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "data", "type": "pdf"}],
                        "output_root": "data/output",
                        "log_dir": "work/logs",
                        "cache_root": "work/cache",
                    },
                    base_dir=base,
                )

    def test_rejects_cache_output_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "image"}],
                        "output_root": "work",
                        "log_dir": "logs",
                        "cache_root": "work/cache",
                    },
                    base_dir=base,
                )

    def test_rejects_log_cache_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "image"}],
                        "output_root": "output",
                        "log_dir": "cache/logs",
                        "cache_root": "cache",
                    },
                    base_dir=base,
                )

    def test_rejects_manifest_inside_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "image"}],
                        "output_root": "output",
                        "log_dir": "logs",
                        "cache_root": "cache",
                        "manifest_path": "cache/manifest.sqlite3",
                    },
                    base_dir=base,
                )

    def test_rejects_home_as_cache_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            with self.assertRaises(ConfigError):
                config_from_mapping(
                    {
                        "input_sources": [{"path": "inputs", "type": "image"}],
                        "output_root": "output",
                        "log_dir": "logs",
                        "cache_root": str(Path.home()),
                    },
                    base_dir=base,
                )

    def test_accepts_gpu_selector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config = config_from_mapping(
                {
                    "input_sources": [{"path": "inputs", "type": "image"}],
                    "output_root": "out",
                    "log_dir": "logs",
                    "cache_root": "cache",
                    "runtime": {"device": "gpu:0"},
                },
                base_dir=base,
            )
            self.assertEqual(config.runtime.device, "gpu:0")


if __name__ == "__main__":
    unittest.main()
