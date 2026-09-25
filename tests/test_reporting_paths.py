"""Exercise the weak-table CLI in an isolated checkout, using saved records."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WeakTablePaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="weak_table_paths_")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.repo = self.work / "arbitrary_checkout_name"
        self.results = self.repo / "results/revision_weak_comparison"
        self.results.mkdir(parents=True)
        self.cwd = self.work / "invoking_directory"
        self.cwd.mkdir()
        self.script = self.repo / "make_revision_weak_table.py"
        shutil.copy2(ROOT / self.script.name, self.script)
        self.raw = self.results / "raw_results.csv"
        shutil.copy2(ROOT / "results/revision_weak_comparison/raw_results.csv", self.raw)
        self.raw_hash = hashlib.sha256(self.raw.read_bytes()).hexdigest()

    def run_table(self, arguments, expected_output):
        completed = subprocess.run(
            [sys.executable, str(self.script), *arguments],
            cwd=self.cwd,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue(expected_output.is_file())
        contents = expected_output.read_bytes()
        self.assertIn(b"\\label{tab:weak_comparison}", contents)
        metadata = json.loads((self.results / "table_manifest.json").read_text())
        self.assertEqual(metadata["output_sha256"], hashlib.sha256(contents).hexdigest())
        self.assertEqual(
            (self.repo.parent / metadata["output"]).resolve(), expected_output.resolve()
        )
        self.assertEqual(hashlib.sha256(self.raw.read_bytes()).hexdigest(), self.raw_hash)
        return contents

    def test_default_stays_inside_checkout_from_another_directory(self):
        self.run_table([], self.repo / "generated_assets/tables/revision_weak_comparison.tex")
        self.assertFalse((self.repo.parent / "latex_source").exists())

    def test_relative_output_and_results_follow_invoking_directory(self):
        output = self.cwd / "reports/weak_table.tex"
        self.run_table(
            ["--results", "../arbitrary_checkout_name/results/revision_weak_comparison",
             "--output", "reports/weak_table.tex"],
            output,
        )

    def test_absolute_external_output_preserves_table_contents(self):
        default = self.repo / "generated_assets/tables/revision_weak_comparison.tex"
        reference = self.run_table([], default)
        output = self.work / "external destination/weak_table.tex"
        actual = self.run_table(["--output", str(output)], output)
        self.assertEqual(reference, actual)


if __name__ == "__main__":
    unittest.main()
