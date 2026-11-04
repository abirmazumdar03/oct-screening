from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def _load_with_includes(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    includes = cfg.pop("include", None) or []
    merged = OmegaConf.create()
    for inc in includes:
        merged = OmegaConf.merge(merged, _load_with_includes(path.parent / inc))
    return OmegaConf.merge(merged, cfg)

def load_config(path, overrides=None) -> DictConfig:
    """Load YAML (with `include:` support) and apply dotlist overrides like ['train.epochs=3']."""
    cfg = _load_with_includes(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg

def to_dict(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
