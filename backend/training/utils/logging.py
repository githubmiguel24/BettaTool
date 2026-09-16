"""Self-contained run directories: `training/runs/<timestamp>_<name>/`.

Each run directory holds the resolved config, the git commit hash, a metric
CSV that is appended to every epoch, and whatever checkpoints/plots the
caller writes into it. This is the only place that decides the run-directory
layout, so `train_hrnet.py`, `evaluate.py`, and `benchmark.py` all import it
rather than re-deriving paths.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit_hash() -> str:
    """Returns the current git commit hash, or 'unknown' if not in a git repo.

    Never raises — thesis code must still produce a run directory when the
    checkout is a plain unzipped folder rather than a git clone.
    """
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
        pass
    return "unknown"


def create_run_dir(runs_root: str | Path, experiment_name: str, resolved_config: dict[str, Any]) -> Path:
    """Creates `<runs_root>/<UTC timestamp>_<experiment_name>/` and seeds it.

    Writes `config.resolved.yaml` (via json for zero extra dependency risk,
    suffixed .json to be explicit) and `git_commit.txt` into the new
    directory immediately, before any training happens, so a run that
    crashes on step 1 still leaves a fully-attributable record.

    Args:
        runs_root: parent directory for all runs (created if missing).
        experiment_name: short slug, e.g. "hrnet_w32_nll".
        resolved_config: the fully-merged config dict to snapshot.

    Returns:
        Path to the newly created run directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.resolved.json").write_text(json.dumps(resolved_config, indent=2, default=str))
    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    return run_dir


class MetricCSVLogger:
    """Appends one row per epoch to `<run_dir>/metrics.csv`, header on first write."""

    def __init__(self, run_dir: str | Path, filename: str = "metrics.csv") -> None:
        self.path = Path(run_dir) / filename
        self._fieldnames: list[str] | None = None
        if self.path.is_file():
            with self.path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def log(self, row: dict[str, Any]) -> None:
        """Appends `row` as a new line, writing the header the first time this is called."""
        is_new_file = self._fieldnames is None
        if is_new_file:
            self._fieldnames = list(row.keys())

        # A later epoch may introduce new keys (e.g. stage-2-only metrics).
        # Rather than silently dropping them, widen the header.
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
