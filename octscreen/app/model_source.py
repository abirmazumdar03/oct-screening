# octscreen/app/model_source.py
"""Find the model package from an ordered list of sources: local folder → zip URL → Hugging Face repo.
The first source that yields a VALID package (contract C5) wins."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from octscreen.common.model_meta import load_meta


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_root(folder):
    """A zip may contain the files directly or inside ONE top-level folder: return where model_meta.json is."""
    folder = Path(folder)
    if (folder / "model_meta.json").exists():
        return folder
    hits = sorted(folder.rglob("model_meta.json"))
    return hits[0].parent if hits else folder


def _from_zip_url(url, cache_dir, sha256=None):
    cache = Path(str(cache_dir))
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / (Path(url.split("?")[0]).name or "model_package.zip")
    if not zpath.exists():
        tmp = zpath.with_suffix(".part")
        print(f"[model] downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(zpath)
    if sha256 and _sha256(zpath) != str(sha256).strip().lower():
        zpath.unlink()
        raise ValueError("sha256 mismatch: download corrupted or wrong file")
    out = cache / zpath.stem
    already_extracted = out.exists() and any(out.rglob("model_meta.json"))
    if not already_extracted:
        if out.exists():
            shutil.rmtree(out)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return _package_root(out)


def _from_hf(repo_id, revision, cache_dir):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, revision=revision or None, cache_dir=str(cache_dir)))


def resolve_package(model_cfg):
    errors = []
    for src in model_cfg.sources:
        kind = str(src.type)
        try:
            if kind == "local" and src.get("path"):
                path = Path(str(src.path))
            elif kind == "url_zip" and src.get("url"):
                path = _from_zip_url(str(src.url), model_cfg.cache_dir, src.get("sha256"))
            elif kind == "hf_repo" and src.get("repo_id"):
                path = _from_hf(str(src.repo_id), src.get("revision"), model_cfg.cache_dir)
            else:
                continue                                   # source not configured → skip silently
            meta = load_meta(path)                         # validates the package
            tag = " (PLACEHOLDER)" if meta.get("placeholder") else ""
            print(f"[model] {kind}: {path} → {meta['package_name']} v{meta['package_version']}{tag}")
            return path
        except Exception as e:
            errors.append(f"{kind}: {e}")
    raise RuntimeError("No usable model package. Tried:\n - " + "\n - ".join(errors or ["no sources configured"]))
