"""Run the existing DAY1 analyses using an explicit raw-data folder."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def run_eda(root: Path, raw_dir: Path) -> None:
    root, raw_dir = Path(root).resolve(), Path(raw_dir).resolve()
    env = os.environ.copy()
    env["ESS_RAW_DIR"] = str(raw_dir)
    scripts = root / "src/eda"
    commands = [
        [sys.executable, str(scripts / "ess_batch_summary.py"), "--data-dir", str(raw_dir), "--out-dir", str(root / "results/eda/summary")],
        [sys.executable, str(scripts / "ess_knee_eda.py"), "--root", str(root)],
        [sys.executable, str(scripts / "ess_deltaq_eda.py")],
        [sys.executable, str(scripts / "ess_current_pattern_eda.py")],
        [sys.executable, str(scripts / "ess_integrate_eda.py")],
    ]
    for command in commands:
        print(f"EDA 실행: {Path(command[1]).name}", flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)
    print(f"DAY1 EDA 저장: {root / 'results/eda'}; 모델 학습 없음")

