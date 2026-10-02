"""Launch the pipeline with a local OpenMP search path on macOS when available."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys


def pipeline_environment():
    env = os.environ.copy()
    if platform.system() == 'Darwin':
        spec = importlib.util.find_spec('sklearn')
        if spec and spec.origin:
            folder = Path(spec.origin).parent / '.dylibs'
            if (folder / 'libomp.dylib').is_file():
                paths = env.get('DYLD_LIBRARY_PATH','').split(os.pathsep)
                if str(folder) not in paths:
                    env['DYLD_LIBRARY_PATH'] = os.pathsep.join([str(folder),*[p for p in paths if p]])
    return env


def ensure_openmp_process():
    """Dynamic-library search paths must be set before the Python process starts."""
    env = pipeline_environment()
    if env.get('DYLD_LIBRARY_PATH') != os.environ.get('DYLD_LIBRARY_PATH'):
        os.execve(sys.executable,[sys.executable,'-m','src.pipeline',*sys.argv[1:]],env)


def run_pipeline(root, arguments):
    """Notebook entry point; stream CLI output and raise on a failed stage."""
    return subprocess.run([sys.executable,'-m','src.pipeline',*[str(a) for a in arguments]],
                          cwd=Path(root),env=pipeline_environment(),check=True)
