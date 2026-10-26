#!/usr/bin/env python3
"""Fetch the deploy checkpoints from the Hugging Face Hub.

The release repo must not contain weights (data/ is gitignored, and 16.5 GiB of
checkpoints does not belong in git history). The deploy set is under 500 MB:
one checkpoint per stage, dominated by the DINOv2-Base ground student.

Nothing downloads silently: the Hub path is used only when the local directory is
incomplete AND SNOWY_WEIGHTS_HUB names a repo. Otherwise this raises, because a
missing checkpoint must never degrade into a randomly-initialised model.

    export SNOWY_WEIGHTS_HUB=garasnote/snowy-weights
    python deploy/weights.py --dest ../models/best
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REQUIRED = ("ground_pl_dinov2b.ckpt", "snow_pl_convnextt_k3.ckpt", "qa_cnn.pt")
OPTIONAL = ("smoother_uni", "smoother_bi", "scaler.json")


def missing(dest: Path) -> list[str]:
    return [f for f in REQUIRED if not (dest / f).exists()]


def ensure_weights(dest, hub_repo: str | None = None, *, quiet: bool = False) -> Path:
    """Return a directory holding the deploy set, downloading it if allowed."""
    dest = Path(dest)
    gap = missing(dest)
    if not gap:
        return dest
    hub_repo = hub_repo or os.environ.get("SNOWY_WEIGHTS_HUB")
    if not hub_repo:
        raise FileNotFoundError(
            f"{dest} is missing {gap} and SNOWY_WEIGHTS_HUB is not set. "
            f"Refusing to continue: a stage without its checkpoint would run random "
            f"weights and still print a plausible number.")
    from huggingface_hub import snapshot_download
    if not quiet:
        print(f"fetching {hub_repo} -> {dest}")
    snapshot_download(repo_id=hub_repo, local_dir=str(dest),
                      allow_patterns=[*REQUIRED, *OPTIONAL, "*.json", "*/*"])
    gap = missing(dest)
    if gap:
        raise FileNotFoundError(f"Hub repo {hub_repo} did not provide {gap}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent.parent
                                          / "models" / "best"))
    ap.add_argument("--hub-repo", default=None)
    args = ap.parse_args(argv)
    dest = ensure_weights(args.dest, args.hub_repo)
    print(f"weights ready in {dest}")
    for f in (*REQUIRED, *OPTIONAL):
        p = dest / f
        print(f"  {'OK ' if p.exists() else '-- '} {f}")


if __name__ == "__main__":
    main()
