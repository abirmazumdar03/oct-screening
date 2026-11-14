# Poddar: build & ship the app in `abirmazumdar03/oct-screening`, step by step

All commands are **PowerShell on Windows**, run from the repo root (`C:\dev\oct-screening`).
Each step ends with a **✅ Done when** check. Paste any error into the chat and get it fixed before moving on.

## What the overlay zip adds to Abir's repo

Nothing of Abir's or Aditya's is replaced, except two docs that are **appended to** (`docs/model_card.md`, `docs/decisions.md`) and the README (yours to finalise).

```
octscreen/app/          NEW  model_source · safety · triage · i18n · inference · render · ui_gradio · cli · heatmap · __main__
octscreen/common/model_meta.py   = an exact copy of Abir's file from branch abir/optimize (identical → no merge conflict)
configs/app.yaml        NEW
configs/i18n/en.yaml, hi.yaml    NEW
scripts/run_app.ps1, make_dev_placeholder.py, make_guard_tests.py   NEW
deploy/Dockerfile, Dockerfile.dockerignore, build_hf_space.py, render.yaml   NEW
tests/test_app_*.py     NEW (4 files)
docs/demo_script.md     NEW
docs/model_card.md      Aditya's §1–2 kept, your §3–8 appended
docs/decisions.md       Abir's line kept, app/guard/threshold/offline sections appended
README.md               final README (keeps the team setup line)
STEP_BY_STEP.md         this file
```

How it fits Abir's code:
- Uses **his** `common/config.py`, `common/preprocess.py` and `common/model_meta.py` (contract C5: `schema_version`, `resize_mode`, `explainer` may be `null`).
- **Heatmaps:** his `octscreen/explain/` files are still empty, so the app has its own Grad-CAM in `octscreen/app/heatmap.py`. It reads `explainer.pt` exactly as his `optimize/package.py` writes it. When he fills `octscreen/explain`, the app switches to his automatically.
- Tested against his baseplate: **21 tests pass** (his 6 + your 15).

---

## Step 1 · Branch + copy the files (10 min)

```powershell
cd C:\dev
git clone https://github.com/abirmazumdar03/oct-screening.git     # skip if you already have it
cd oct-screening
git checkout main ; git pull
git checkout -b poddar/app
```

Unzip `poddar-app-overlay.zip` **into the repo folder** (`C:\dev\oct-screening`), say **Yes to replace** for the 3 docs.
Or apply the patch instead: `git am C:\path\to\poddar-app.patch`. That does the copy and the commit in one go; then skip the `git add/commit` line below.

```powershell
git status                              # new files under octscreen/app, configs, scripts, deploy, tests, docs
git add -A ; git commit -m "app: core, UI, CLI, tests, deploy, docs (Poddar)"
```

✅ **Done when** `git log -1` shows your commit on `poddar/app`.

---

## Step 2 · Install the app packages (10–15 min, internet)

```powershell
uv add --group app gradio onnxruntime huggingface_hub
uv add --group app-heatmap torch torchvision timm
uv sync --group app --group app-heatmap
uv run python -c "import gradio, onnxruntime, torch; print('OK', gradio.__version__, onnxruntime.__version__)"
git add pyproject.toml uv.lock ; git commit -m "app: dependency groups"
```

✅ **Done when** it prints `OK` and versions.

---

## Step 3 · Model package + sample scans (5 min)

**Model package:** ask Abir for his placeholder zip (`PLACEHOLDER-random-weights-v0.1.0.zip`, made by `tools/make_placeholder.py` on his branch; it includes `explainer.pt`, so heatmaps work). Unzip it:

```powershell
Expand-Archive .\PLACEHOLDER-random-weights-v0.1.0.zip -DestinationPath artifacts\model_package -Force
```

Not with you yet? Make a tiny one (no heatmap, passes Abir's validator):
```powershell
uv run python scripts/make_dev_placeholder.py
```

**Sample scans:** from Aditya (`python -m octscreen.data.export_packs --pack sample` on Kaggle → `sample_scans` folder). Put it at `artifacts\packs\sample_scans\` with sub-folders `CNV DME DRUSEN NORMAL`.

✅ **Done when** `artifacts\model_package\model_meta.json` exists (not one folder deeper) and `artifacts\packs\sample_scans\CNV\` has images.

---

## Step 4 · Tests (2 min)

```powershell
uv run pytest -q
```

✅ **Done when** all pass (≈ 21; a few skips are OK only if Step 3 isn't done).

---

## Step 5 · CLI (5 min)

```powershell
uv run python -m octscreen.app.cli --images artifacts/packs/sample_scans --lang hi
```

✅ **Done when** the summary says `rejected by guard: 0/N`. Predictions are random with a placeholder, which is expected.

---

## Step 6 · The app (15 min)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1          # -Lang hi · -NoHeatmap · -Share
```

Tick each:
- [ ] Yellow PLACEHOLDER banner visible
- [ ] Clicking a sample scan analyses it; triage line is first and biggest
- [ ] English ↔ हिन्दी switches the result, bars and About panel (time doesn't change = no re-run)
- [ ] A selfie / colour photo is refused
- [ ] Low confidence → ⚪ "refer for manual review"
- [ ] With Abir's placeholder: the heatmap panel appears. With `-NoHeatmap`: it's hidden and everything else works

Look at the console when the app starts: `[heatmap] Grad-CAM ready (...)` means heatmaps are on; `[heatmap] ... → heatmaps off` tells you why not.

```powershell
git push -u origin poddar/app        # then open a Pull Request on GitHub → ask Abir to review
```

✅ **Done when** every box is ticked and the PR is open.

---

## Step 7 · Guard test set (20 min)

```powershell
uv run python scripts/make_guard_tests.py
# + copy ~5 real phone photos (desk, face, document, screenshot, a black-and-white photo) into artifacts\guard_tests\
uv run python -m octscreen.app.cli --images artifacts/guard_tests --out artifacts/guard_tests_results.csv
```

✅ **Done when** both counts (sample scans rejected 0/N · non-OCT rejected X/Y) are written in `docs/decisions.md` → *Guard*.

---

## Step 8 · Docker (30–40 min, mostly waiting; Docker Desktop running)

```powershell
docker build -f deploy/Dockerfile --build-arg WITH_HEATMAP=0 -t oct-screen:lite .
docker run --rm -p 7860:7860 -v "${PWD}\artifacts\model_package:/app/model_package:ro" oct-screen:lite
# open http://localhost:7860 → works? Ctrl+C, then the full one:
docker build -f deploy/Dockerfile -t oct-screen .
docker run --rm -p 7860:7860 -v "${PWD}\artifacts\model_package:/app/model_package:ro" oct-screen
```

Fixes: `ghcr.io` pull fails → replace the `COPY --from=ghcr.io/...` line with `RUN pip install uv`. `--frozen` fails → `uv lock`, commit `uv.lock`, rebuild.

✅ **Done when** both serve the app on `localhost:7860`.

---

## Step 9 · Host #1: Hugging Face Space (20 min)

```powershell
uv run hf auth login               # your own WRITE token from huggingface.co/settings/tokens
uv run python deploy/build_hf_space.py --package artifacts/model_package                       # stage, look at artifacts\deploy\hf_space
uv run python deploy/build_hf_space.py --package artifacts/model_package --push --space <your-hf-username>/oct-screening
```

(Use `oct-screening-team/oct-screening` only if that HF organisation exists and you're in it.) Watch the Space's *Logs* tab, ~3–8 min.

✅ **Done when** the Space opens, shows the online banner, and a sample scan gives a result.

---

## Step 10 · Host #2: Render (20 min)

1. Abir uploads the package zip to a **GitHub Release** of this repo and sends you the asset URL and its `.sha256`.
2. Render → **New → Web Service** → connect `abirmazumdar03/oct-screening` → Runtime **Docker**, Dockerfile path `./deploy/Dockerfile`, Docker context `.`, plan **Free**.
3. Environment variables: `OCT_ENABLE_HEATMAP=0`, `OCT_ONLINE_DEMO=1`, `OCT_MODEL_URL=<release asset URL>`, `OCT_MODEL_SHA256=<sha256>`.
4. Optional: Docker build argument `WITH_HEATMAP=0` (smaller, faster build).

(Alternative: **New → Blueprint** with `deploy/render.yaml`. Render looks for `render.yaml` at the repo root, so copy it there or set the blueprint path if the form allows.)

✅ **Done when** the Render URL opens (first load after sleep ≈ 1 min) and gives a result.

---

## Step 11 · Threshold (tonight, when Abir's threshold table is ready)

Rule: the **lowest** threshold with *silent urgent misses ≤ 0.5 %* while *flagged uncertain ≤ 10 %* (if impossible, prefer fewer misses).
Set `policy.confidence_threshold` in `configs/app.yaml`, delete the word `PROVISIONAL`, add one line with the evidence in `docs/decisions.md`, then `uv run pytest -q`.

## Step 12 · Real model swap (tonight → Sat 1 PM for v1.0.1)

```powershell
Get-FileHash .\oct-student-kd-v1.0.0.zip -Algorithm SHA256        # must match the .sha256 file
Remove-Item -Recurse -Force artifacts\model_package
Expand-Archive .\oct-student-kd-v1.0.0.zip -DestinationPath artifacts\model_package
uv run pytest -q
uv run python -m octscreen.app.cli --images artifacts/packs/sample_scans          # most predictions now match folder names
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1                       # PLACEHOLDER banner must be GONE
uv run python deploy/build_hf_space.py --package artifacts/model_package --push --space <your-hf-username>/oct-screening
# Render: update OCT_MODEL_URL + OCT_MODEL_SHA256 → Manual Deploy
```

## Step 13 · Offline test on 3 laptops (Sat morning)

On each laptop: `git pull ; uv sync --all-groups`, copy package + sample scans, **Wi-Fi OFF**, `scripts\run_app.ps1`.
Check: opens by itself · 4 classes · Hindi · selfie/tiny refused · time vs Abir's benchmark · close + reopen works.
Fill the table in `docs/decisions.md`. At least **2 laptops** must pass.

## Step 14 · Docs, video, freeze (Sat → Sun)

README links + benchmark table + screenshot (search `TODO`/`TBD`) · model card §3–6 numbers from Abir · rehearse `docs/demo_script.md` twice · record backup video Sat · USB stick ·
**Sat 8 PM:** `git checkout main ; git pull ; uv run pytest -q ; git tag v1.0.0-demo ; git push origin v1.0.0-demo`.

---

## Merge notes (for the PR)

- `pyproject.toml` / `uv.lock` will conflict with Abir's `abir/optimize` branch (he filled the `optimize` group, you filled `app`). Fix: keep **both** groups in `pyproject.toml`, then run `uv lock` and commit the new `uv.lock`.
- `octscreen/common/model_meta.py` is identical to Abir's, so it won't conflict.
- If Abir merges first, just: `git checkout poddar/app ; git merge main`, fix pyproject as above, `uv lock`, `uv run pytest -q`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Invalid model package ...: missing key: schema_version` | old/wrong package; use Abir's zip or `scripts/make_dev_placeholder.py` |
| `No usable model package` + `model_meta.json` not found | unzipped one folder too deep; move files up into `artifacts\model_package` |
| `Invalid app policy: policy.triage has no entry for [...]` | the model has a new class; add it to `configs/app.yaml → policy.triage` |
| `classifier outputs N scores but model_meta.json lists M classes` | package/meta mismatch → tell Abir |
| `[heatmap] ... → heatmaps off` | no `explainer.pt`, torch not installed, or load error (message says which); predictions still work |
| `running scripts is disabled` | you forgot `-ExecutionPolicy Bypass` |
| port 7860 busy | close the other app, or `$env:PORT=7861` before `run_app.ps1` |
| Gradio API error | `uv add --group app "gradio<6"` |
