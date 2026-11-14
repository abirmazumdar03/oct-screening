# octscreen/app/render.py
"""Turn a screening result into language-specific text. No UI imports, so it's reusable by any front-end."""


def _pct(v):
    return f"{v:.0%}"


def result_markdown(res, lang, tr, threshold):
    if not res:
        return ""
    out = []
    if res.get("placeholder"):
        out.append(f"> {tr.t(lang, 'placeholder_banner')}")
    if res["status"] == "rejected":
        out.append(f"### {tr.t(lang, 'rejected_heading')}")
        out += [f"- {tr.t(lang, 'guard.' + code, **params)}" for code, params in res["issues"]]
        out.append(f"---\n{tr.t(lang, 'disclaimer')}")
        return "\n\n".join(out)
    c = res["class_name"]
    out.append(f"## {tr.t(lang, 'triage.' + res['level'])}")
    out.append(f"**{tr.t(lang, 'prediction')}:** {tr.t(lang, f'class.{c}.name', default=c)} · {_pct(res['confidence'])}")
    out.append(f"_{tr.t(lang, f'class.{c}.desc', default='')}_")
    if res["uncertain"]:
        out.append(tr.t(lang, "uncertain_msg", conf=_pct(res["confidence"]), thr=_pct(threshold)))
    out += [f"⚠ {tr.t(lang, 'guard.' + code, **params)}" for code, params in res["issues"]]   # guard mode "warn"
    out.append(tr.t(lang, "time_label", ms=res["model_ms"]))
    if "heatmap_ms" in res:
        out.append(tr.t(lang, "heatmap_time", ms=res["heatmap_ms"]))
    out.append(f"---\n{tr.t(lang, 'disclaimer')}")
    return "\n\n".join(out)


def label_dict(res, lang, tr):
    """{localised class name: probability} for a bar-chart widget; None if nothing to show."""
    if not res or res["status"] != "ok":
        return None
    return {tr.t(lang, f"class.{c}.short", default=c): p for c, p in res["probs"].items()}


def about_markdown(meta, lang, tr):
    m = meta.get("metrics") or {}
    na = tr.t(lang, "about.not_measured")

    def pct(v):
        return na if v is None else _pct(v)

    lat = na if m.get("latency_ms_p50") is None else f"{m['latency_ms_p50']} ms · {m.get('latency_device') or ''}"
    rows = [
        ("about.package", f"{meta['package_name']} v{meta['package_version']}"),
        ("about.precision", meta["classifier"]["precision"].upper()),
        ("about.size", na if m.get("size_mb") is None else f"{m['size_mb']} MB"),
        ("about.accuracy", pct(m.get("test_accuracy"))),
        ("about.macro_f1", pct(m.get("test_macro_f1"))),
        ("about.urgent", pct(m.get("urgent_sensitivity"))),
        ("about.latency", lat),
    ]
    lines = [f"- **{tr.t(lang, k)}:** {v}" for k, v in rows]
    if meta.get("placeholder"):
        lines.insert(0, f"> {tr.t(lang, 'placeholder_banner')}")
    return "\n".join(lines) + f"\n\n{tr.t(lang, 'about.data')}"
