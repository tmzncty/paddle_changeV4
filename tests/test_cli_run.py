import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.orchestrator import ProjectItemResult, ProjectRunResult


class ProjectRunCliTests(unittest.TestCase):
    def _write_config(self, root: Path, source: Path) -> Path:
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "input_sources": [{"path": str(source), "type": "pdf"}],
                    "output_root": str(root / "work" / "output"),
                    "log_dir": str(root / "work" / "logs"),
                    "cache_root": str(root / "work" / "cache"),
                    "runtime": {"device": "cpu", "ocr_workers": 2},
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_run_json_emits_machine_readable_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            source.write_bytes(b"%PDF-fake")
            config_path = self._write_config(root, source)

            final_pdf = (
                root
                / "work"
                / "output"
                / "source-001"
                / "pdf"
                / "source"
                / "searchable.pdf"
            )
            fake_result = ProjectRunResult(
                items=(
                    ProjectItemResult(
                        source=source.resolve(),
                        kind="pdf",
                        status="success",
                        pages_dir=final_pdf.parent / "pages",
                        ocr_dir=final_pdf.parent / "ocr",
                        searchable_pdf=final_pdf,
                    ),
                )
            )

            stdout = io.StringIO()
            with patch(
                "paddle_batch_ocr.cli.run_project",
                return_value=fake_result,
            ) as run:
                with contextlib.redirect_stdout(stdout):
                    rc = main(
                        [
                            "run",
                            "--config",
                            str(config_path),
                            "--dpi",
                            "200",
                            "--json",
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(run.call_args.kwargs["dpi"], 200)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["success"], 1)
            self.assertEqual(payload["failed"], 0)
            self.assertEqual(payload["items"][0]["kind"], "pdf")
            self.assertEqual(
                payload["items"][0]["searchable_pdf"],
                str(final_pdf),
            )

    def test_run_returns_one_when_any_item_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            source.write_bytes(b"%PDF-fake")
            config_path = self._write_config(root, source)
            fake_result = ProjectRunResult(
                items=(
                    ProjectItemResult(
                        source=source.resolve(),
                        kind="pdf",
                        status="failed",
                        error="ProjectRunError: boom",
                    ),
                )
            )

            with patch(
                "paddle_batch_ocr.cli.run_project",
                return_value=fake_result,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = main(["run", "--config", str(config_path)])

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
