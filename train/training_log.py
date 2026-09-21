"""Console mirroring and wall-clock timing for training runs (stdlib only)."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback


class _Tee:
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class TrainingLog:
    def __init__(self):
        self.steps = 0
        self.training_seconds = 0.0
        self.checkpoint_seconds = 0.0

    @contextmanager
    def phase(self, name, synchronize=None):
        # Synchronize only at phase boundaries, not on every training step.
        if synchronize:
            synchronize()
        started = time.perf_counter()
        print(f"[time] {name}: started at {datetime.now().astimezone().isoformat(timespec='seconds')}")
        try:
            yield
        finally:
            try:
                if synchronize:
                    synchronize()
            finally:
                elapsed = time.perf_counter() - started
                if name == "training":
                    self.training_seconds += elapsed
                elif name.startswith("checkpoint-"):
                    self.checkpoint_seconds += elapsed
                print(f"[time] {name}: {elapsed:.3f} s")


@contextmanager
def record_training(args, log_dir):
    started = time.perf_counter()
    start_time = datetime.now().astimezone()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    path = log_dir / f"{args.stage}_{start_time:%Y%m%d_%H%M%S_%f}_rank{rank}_pid{os.getpid()}.log"
    run = TrainingLog()
    with path.open("x", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            print(f"Log file: {path.resolve()}")
            print(f"Start time: {start_time.isoformat(timespec='seconds')}")
            print(f"Command arguments: {json.dumps(sys.argv, ensure_ascii=False)}")
            print(f"Working directory: {Path.cwd()}")
            print("Run parameters: " + json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))
            print(f"Dataset root: {args.data_dir.resolve()}")
            status = "completed"
            try:
                yield run
            except BaseException:
                status = "failed/interrupted"
                traceback.print_exc()
                raise
            finally:
                print(f"End time: {datetime.now().astimezone().isoformat(timespec='seconds')} ({status})")
                print(f"Total elapsed: {time.perf_counter() - started:.3f} s")
                print(f"Completed optimizer steps: {run.steps}")
                print(f"Training elapsed: {run.training_seconds:.3f} s (includes data loading and periodic checkpoints)")
                print(f"Checkpoint elapsed: {run.checkpoint_seconds:.3f} s")
                average = f"{run.training_seconds / run.steps:.6f} s" if run.steps else "N/A (no completed steps)"
                print(f"Average per optimizer step: {average}")
