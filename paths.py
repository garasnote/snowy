"""
Central configuration for data/output locations that live outside this repo.

Every value here can be overridden with an environment variable; otherwise it
falls back to a local, relative directory, so the code runs out of the box
without any specific cluster, SLURM, or account layout.

    export SNOW_DATA_DIR=/path/to/data
    export SNOW_WORKSPACE_DIR=/path/to/large/intermediate/outputs
    export SNOW_MODEL_CACHE_DIR=/path/to/cached/backbone/weights

Scripts import from this module instead of hardcoding paths:

    from paths import DATA_DIR
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("SNOW_DATA_DIR", REPO_ROOT / "data"))
WORKSPACE_DIR = Path(os.environ.get("SNOW_WORKSPACE_DIR", REPO_ROOT / "workspace"))
MODEL_CACHE_DIR = Path(os.environ.get("SNOW_MODEL_CACHE_DIR", REPO_ROOT / "model_cache"))
ANALYSIS_DIR = REPO_ROOT / "analysis"
