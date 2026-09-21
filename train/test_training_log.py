"""Run with python -m unittest train.test_training_log."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from train.training_log import record_training


class TrainingLogTests(unittest.TestCase):
    def test_console_mirroring_and_step_average(self):
        console = io.StringIO()
        args = argparse.Namespace(stage="codec", data_dir=Path("dataset"))
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(console), redirect_stderr(console):
                with patch("train.training_log.time.perf_counter", side_effect=[0, 1, 7, 9]):
                    with record_training(args, Path(directory) / "logs") as run:
                        with run.phase("training"):
                            print("loss=0.125")
                            print("warning example", file=sys.stderr)
                            run.steps = 3
                self.assertIs(sys.stdout, console)
                self.assertIs(sys.stderr, console)
            logs = list((Path(directory) / "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertEqual(text, console.getvalue())
            for expected in ("Run parameters:", "Dataset root:", "loss=0.125", "warning example",
                             "Start time:", "End time:", "Total elapsed: 9.000 s",
                             "Average per optimizer step: 2.000000 s"):
                self.assertIn(expected, text)

    def test_failure_is_logged_and_reraised(self):
        args = argparse.Namespace(stage="flow", data_dir=Path("dataset"))
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with record_training(args, directory) as run:
                        with run.phase("model loading"):
                            raise RuntimeError("test failure")
            text = next(Path(directory).glob("*.log")).read_text(encoding="utf-8")
            for expected in ("RuntimeError: test failure", "failed/interrupted", "[time] model loading:",
                             "Average per optimizer step: N/A", "End time:"):
                self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
