#!/usr/bin/env python3
"""Publish the deploy set to the Hugging Face Hub and stand up the Space.

Run this on a machine with network access (the agent's container has none): it is
the only part of the deployment that cannot be done from inside the sandbox.

What it does, in one command:
  1. checks you are logged in (token is read from your local HF login, never printed)
  2. creates a PRIVATE model repo and uploads the deploy checkpoints (~33 MiB)
  3. creates a PRIVATE Space and uploads the bundle from deploy/pack_space.py
  4. sets the Space's HF_TOKEN secret so the private Space can read the private
     weights repo at boot, and SNOWY_WEIGHTS_HUB so it knows where to look

Nothing is made public. Re-running is safe: existing repos are reused.

    python deploy/publish_hub.py --user garasnote
    python deploy/publish_hub.py --user garasnote --weights ../models/best \
        --space-dir deploy/space_bundle --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=None,
                    help="Hugging Face username or org (default: the account you are "
                         "logged in as)")
    ap.add_argument("--weights", default=str(REPO.parent / "models" / "best"),
                    help="directory holding the deploy checkpoints")
    ap.add_argument("--weights-repo", default="snowy-weights")
    ap.add_argument("--space-repo", default="snowy-demo")
    ap.add_argument("--space-dir", default=str(REPO / "deploy" / "space_bundle"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the files, upload nothing")
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install -U huggingface_hub")
        return 2

    weights_dir = Path(args.weights).resolve()
    space_dir = Path(args.space_dir).resolve()
    required = ("ground_mobilenet_0.916.ckpt", "snow_mobilenet_0.821.ckpt", "qa_cnn.pt")
    missing = [f for f in required if not (weights_dir / f).exists()]
    if missing:
        print(f"refusing to publish: {weights_dir} is missing {missing}")
        return 2
    if not (space_dir / "app.py").exists():
        print(f"refusing to publish: no Space bundle at {space_dir}. Run:\n"
              f"    python deploy/pack_space.py --dest {space_dir} --hub-repo {args.user}/"
              f"{args.weights_repo}")
        return 2

    token = get_token()
    if not token:
        print("not logged in. Run:  hf auth login   (create a write token at "
              "https://huggingface.co/settings/tokens)")
        return 2
    api = HfApi()

    who = api.whoami(token=token)
    print(f"authenticated as: {who.get('name')}")
    user = args.user or who.get("name")
    if not user:
        print("could not determine your account; pass --user")
        return 2
    weights_repo_id = f"{user}/{args.weights_repo}"
    space_repo_id = f"{user}/{args.space_repo}"

    # only the deploy set: models/best also holds a 378 MB experiment checkpoint that
    # has nothing to do with deployment
    KEEP = ("ground_mobilenet_0.916.ckpt", "snow_mobilenet_0.821.ckpt", "qa_cnn.pt",
            "scaler.json", "qa_gradboost_lite.pkl",
            "smoother_uni/smoother_gru_uni_best.ckpt", "smoother_uni/scaler.json",
            "smoother_gru_bi_best.ckpt", "smoother_gru_bi_best_scaler.json")
    files = [weights_dir / f for f in KEEP if (weights_dir / f).is_file()]
    skipped = sorted(p for p in weights_dir.rglob("*")
                     if p.is_file() and str(p.relative_to(weights_dir)) not in KEEP)
    total = sum(p.stat().st_size for p in files)
    print(f"\nweights  {weights_repo_id}  (private)  {len(files)} files, "
          f"{total/1024/1024:.1f} MiB")
    for p in files:
        print(f"    {p.relative_to(weights_dir)}  {p.stat().st_size/1024:.0f} KiB")
    if skipped:
        print(f"  not uploaded ({len(skipped)} file(s), "
              f"{sum(p.stat().st_size for p in skipped)/1024/1024:.0f} MiB), not part of "
              f"the deploy set:")
        for p in skipped[:6]:
            print(f"    {p.relative_to(weights_dir)}")
    print(f"\nspace    {space_repo_id}  (private)  from {space_dir}")
    for p in sorted(space_dir.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(space_dir)}")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    api.create_repo(weights_repo_id, repo_type="model", private=True, exist_ok=True)
    for p in files:
        rel = str(p.relative_to(weights_dir))
        api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                        repo_id=weights_repo_id, repo_type="model",
                        commit_message=f"deploy set: {rel}")
    print(f"uploaded weights -> https://huggingface.co/{weights_repo_id}")

    try:
        api.create_repo(space_repo_id, repo_type="space", space_sdk="gradio",
                        private=True, exist_ok=True)
    except Exception as exc:
        msg = str(exc)
        print(f"\ncould not create the Space: {type(exc).__name__}: {msg[:300]}")
        if "402" in msg or "Payment Required" in msg:
            print("""
Hugging Face now charges for Gradio/Docker Spaces on free cpu-basic: only Static
Spaces (HTML/JS, no Python) are free on an account without PRO, and a static Space
cannot run torch. So the Space part of this deployment needs either
  * a PRO subscription (~$9/month) -> then re-run this script, or
  * a different free host for the Gradio app; the app itself is unchanged.
The WEIGHTS ARE UPLOADED AND USABLE either way -- that part is done.""")
        else:
            print("Re-run this script once the cause is fixed; existing repos are reused.")
        return 1
    api.upload_folder(repo_id=space_repo_id, repo_type="space", folder_path=str(space_dir),
                      commit_message="snowy deployment demo")
    # the private Space must be able to read the private weights repo: a secret for
    # the token, and a plain variable telling weights.py where to look
    try:
        api.add_space_variable(repo_id=space_repo_id, key="SNOWY_WEIGHTS_HUB",
                               value=weights_repo_id)
        print(f"set Space variable SNOWY_WEIGHTS_HUB={weights_repo_id}")
    except Exception as exc:
        print(f"could not set the SNOWY_WEIGHTS_HUB variable ({type(exc).__name__}: {exc}).")
    try:
        api.add_space_secret(repo_id=space_repo_id, key="HF_TOKEN", value=token)
        print("set Space secret HF_TOKEN (value never printed)")
    except Exception as exc:
        print(f"could not set the HF_TOKEN secret automatically ({type(exc).__name__}: {exc}).")
        print(f"Set it by hand: https://huggingface.co/spaces/{space_repo_id}/settings "
              f"-> Variables and secrets -> New secret, name HF_TOKEN, value = your token.")
    print(f"uploaded Space  -> https://huggingface.co/spaces/{space_repo_id}")
    print("\nCheck the build log on the Space page. First boot downloads the weights "
          "from the private repo and pre-caches the backbones; expect a few minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
