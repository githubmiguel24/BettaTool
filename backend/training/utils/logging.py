# handles setting up training run folders, saving configs, and logging epoch metrics to csv

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# grab current git commit hash or just return unknown if not in a repo
def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # dont crash if this is just an unzipped folder without git
        pass
    return "unknown"


# make the timestamped run directry and dump initial config + commit info right away
def create_run_dir(runs_root: str | Path, experiment_name: str, resolved_config: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # save config and git hash before training starts in case it crashes early
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2, default=str))
    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    return run_dir


# simple csv logger that appends metrics every epoch
class MetricCSVLogger:

    # load existing CSV header if the file is already there
    def __init__(self, run_dir: str | Path, filename: str = "metrics.csv") -> None:
        self.path = Path(run_dir) / filename
        self._fieldnames: list[str] | None = None
        if self.path.is_file():
            with self.path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    # append a row of metrics and create the file header on first write
    def log(self, row: dict[str, Any]) -> None:
        is_new_file = self._fieldnames is None
        if is_new_file:
            self._fieldnames = list(row.keys())

        # expand the headr if new metric keys show up in later stages
        missing = [k for k in row.keys() if k not in self._fieldnames]
        if missing:
            self._fieldnames.extend(missing)
            self._rewrite_header_if_needed()

        write_header = is_new_file or not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    # rewrite the whole csv when new columns get added mid-run
    def _rewrite_header_if_needed(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in self._fieldnames})