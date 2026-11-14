# deploy/build_hf_space.py
"""Stage a Hugging Face Space (Gradio SDK) and optionally upload it.

Usage:
  uv run python deploy/build_hf_space.py --package artifacts/model_package                            # stage only
  uv run python deploy/build_hf_space.py --package artifacts/model_package --push --space oct-screening-team/oct-screening
Needs `hf auth login` first (your own write token), or the HF_TOKEN environment variable.
"""
import argparse
import importlib.metadata as md
import json
import shutil
import subprocess
from pathlib import Path

APP_PY = '''import os
os.environ.setdefault("OCT_MODEL_DIR", "model_package")
os.environ.setdefault("OCT_HOST", "0.0.0.0")
os.environ.setdefault("OCT_ONLINE_DEMO", "1")
from octscreen.app.ui_gradio import main
main(["--set", "ui.examples_dir=examples", "ui.open_browser=false"])
'''

README = """---
title: OCT Retina Screening
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: OCT screening research prototype (EN/HI)
---

# OCT Retina Screening Assistant (research prototype)

Screening aid only, not a diagnosis. **Please use the sample scans and do not upload real patient data.**
Model package: {package}. Data: Kermany et al., *Cell* 2018 (CC BY 4.0). Code: {repo}
"""

# HF installs Gradio itself from `sdk_version`; listing it again in requirements.txt can clash.
SDK_PROVIDED = ("gradio==", "gradio-client==")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="artifacts/model_package")
    ap.add_argument("--examples", default="artifacts/packs/sample_scans")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="artifacts/deploy/hf_space")
    ap.add_argument("--no-heatmap", action="store_true", help="smaller build without torch")
    ap.add_argument("--repo-url", default="https://github.com/abirmazumdar03/oct-screening")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--space", default=None, help="e.g. oct-screening-team/oct-screening")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    (out / "octscreen").mkdir(parents=True)
    shutil.copy2("octscreen/__init__.py", out / "octscreen/__init__.py")
    for sub in ("common", "app", "explain"):                       # only what the app needs
        shutil.copytree(f"octscreen/{sub}", out / "octscreen" / sub, ignore=ignore)
    (out / "configs").mkdir()
    shutil.copy2("configs/app.yaml", out / "configs/app.yaml")
    shutil.copytree("configs/i18n", out / "configs/i18n")
    shutil.copytree(a.package, out / "model_package")
    if Path(a.examples).is_dir():
        for cls_dir in sorted(p for p in Path(a.examples).iterdir() if p.is_dir()):
            for img in sorted(cls_dir.glob("*"))[: a.per_class]:
                (out / "examples" / cls_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out / "examples" / cls_dir.name / img.name)
    (out / "app.py").write_text(APP_PY, encoding="utf-8")

    groups = ["--group", "app"] + ([] if a.no_heatmap else ["--group", "app-heatmap"])
    reqs = subprocess.run(["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project", *groups],
                          capture_output=True, text=True, check=True).stdout
    reqs = "\n".join(line for line in reqs.splitlines() if not line.strip().lower().startswith(SDK_PROVIDED))
    (out / "requirements.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" + reqs + "\n",
                                          encoding="utf-8")

    meta = json.loads((out / "model_package/model_meta.json").read_text(encoding="utf-8"))
    pkg = f"{meta['package_name']} v{meta['package_version']}" + (" (PLACEHOLDER)" if meta.get("placeholder") else "")
    (out / "README.md").write_text(README.format(gradio=md.version("gradio"), package=pkg, repo=a.repo_url),
                                   encoding="utf-8")
    print(f"[staged] {out}  ({pkg})")

    if a.push:
        if not a.space:
            raise SystemExit("--space <org>/<name> is required with --push")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True, private=a.private)
        api.upload_folder(folder_path=str(out), repo_id=a.space, repo_type="space",
                          commit_message=f"Deploy {pkg}", delete_patterns="*")
        print(f"[pushed] https://huggingface.co/spaces/{a.space}  (build takes ~3–8 min)")


if __name__ == "__main__":
    main()
