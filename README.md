# OCT Retina Screening Assistant (ZEISS BeyondCode'26, AFI theme)

Small, offline AI for screening retinal OCT scans on an ordinary laptop.

> **Screening aid for research/demo only, not a diagnosis.** | [Online demo](https://huggingface.co/spaces/oct-screening-team/oct-screening) · [Backup demo](https://oct-screening.onrender.com) <!-- TODO Sat: paste the real links -->

## What it does (30 s)

A clinic operator uploads a retinal OCT scan and, in well under a second, gets a clear answer in **English or हिन्दी**:

1. a **triage instruction** first and biggest: 🔴 urgent referral · 🟡 routine follow-up · 🟢 no signs of disease · ⚪ uncertain, refer for manual review
2. the predicted condition (CNV, DME, Drusen, Normal) with **confidence bars**
3. a **heatmap** of where the model looked (when PyTorch is installed)
4. the **time taken** on this computer's processor

It runs with **no internet and no GPU**, refuses images that are obviously not OCT scans, and never shows a disease verdict on its own when it isn't confident.

<!-- TODO Sat: screenshot of the app in EN and HI side by side → docs/img/app_en_hi.png -->

## Results (measured)

<!-- TODO Sat: paste the benchmark table from docs/benchmark.md (Abir). Real numbers only, machines named. -->
| Machine | Model | Size | Median latency (1 scan) | Test macro-F1 | Urgent sensitivity |
|---|---|---|---|---|---|
| Abir's i3 laptop (no GPU) | INT8 student (ONNX) | _TBD_ MB | _TBD_ ms | _TBD_ | _TBD_ |
| Poddar's i5 laptop (no GPU) | INT8 student (ONNX) | _TBD_ MB | _TBD_ ms | (same) | (same) |

## Run it on Windows in 5 commands

```powershell
# 1. install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 2. get the code
git clone https://github.com/abirmazumdar03/oct-screening.git ; cd oct-screening
# 3. install the app (internet needed once)
uv sync --group app --group app-heatmap
# 4. unzip the model package into artifacts\model_package  (and sample scans into artifacts\packs\sample_scans)
# 5. start it: works with Wi-Fi OFF from here on
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1          # -Lang hi · -NoHeatmap · -Share
```

No browser? Screen a whole folder from the terminal:
`uv run python -m octscreen.app.cli --images artifacts/packs/sample_scans --lang hi` → table + CSV.

## How it works

```
Kermany OCT data → teacher model → knowledge distillation → small student → INT8 quantisation → ONNX
                                                                                               │
          configs/app.yaml (policy) ──┐                                                        ▼
 scan → safety guard → preprocess → ONNX Runtime (CPU) → softmax → threshold + triage → EN/HI text
                                  └→ Grad-CAM on the FP32 twin (optional) → heatmap overlay
```

- The app reads only the **model package** (`model_meta.json` + `classifier.onnx` + optional `explainer.pt`). Class names, image size and temperature come from the package, never from code.
- The package is found from an ordered list of sources: **local folder → zip URL (GitHub Release, sha256-checked) → Hugging Face repo**. The first valid one wins.
- All logic is UI-independent (`octscreen/app/inference.py`, `render.py`); `ui_gradio.py` is the only Gradio file.

## Safety & limitations

- **Input guard (heuristic):** rejects colour photos, images smaller than 128 px, and shapes no OCT scan has. A grayscale image that isn't an OCT scan can still pass. That's why there's a second barrier:
- **Confidence threshold:** below it, the app always shows ⚪ *refer for manual review*. The value was chosen from a validation-set table of "flagged uncertain" vs "silently missed urgent cases" (rule and value in `docs/decisions.md`).
- **Data:** public research data (Kermany et al., *Cell* 2018), mostly from one scanner brand. **Not clinically validated.** Other scanners need validation first.
- **Privacy:** the offline laptop is the intended use, and nothing leaves the machine. The online demo asks users to use the sample scans and never upload patient data.

See [`docs/model_card.md`](docs/model_card.md) for intended use, evaluation and ethical considerations.

## Project structure & docs

```
octscreen/app/      the app: model_source · safety · triage · i18n · inference · render · ui_gradio · cli
octscreen/common/   shared contracts (config, model package, preprocessing)
octscreen/explain/  optional Grad-CAM heatmaps
configs/app.yaml    all app policy (sources, threshold, triage, guard, UI)
configs/i18n/       en.yaml, hi.yaml (add a language = one YAML + one config line)
deploy/             Dockerfile, HF Space builder, Render blueprint
scripts/run_app.ps1 the one command for the demo
docs/               model_card · demo_script · decisions · benchmark
```

## Team setup (developers)

```powershell
uv sync --group train        # Aditya  |  --group optimize (Abir)  |  --group app --group app-heatmap (Poddar)
uv run pytest -q
```

## Data & licence

Data: Kermany, Goldbaum et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", *Cell* 2018 (CC BY 4.0). Code: MIT.

## Team

Aditya (data & training) · Abir (optimisation, evaluation & baseplate) · Poddar (app, safety & delivery)
