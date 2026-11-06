# Demo script (≈ 3 minutes, rehearse twice)

## Before the visit (T-30 min)

- [ ] Primary laptop (Poddar) charged + charger; backup laptop (Aditya) tested Sat
- [ ] `scripts\run_app.ps1` started **once with Wi-Fi off**: browser opens, placeholder banner **absent**
- [ ] Open the HF Space link once (wakes it up), and the Render link (cold start ≈ 1 min)
- [ ] Demo files on the desktop in `demo\`: `1_CNV.jpeg`, `2_NORMAL.jpeg`, `3_selfie.jpg`, `4_borderline.jpeg`
      (find the borderline one beforehand: `uv run python -m octscreen.app.cli --images artifacts/packs/sample_scans`
      → pick a scan whose triage is ⚪ or whose confidence is just above the threshold)
- [ ] Backup video on the desktop + USB stick
- [ ] Browser zoom 125–150 % so the projector shows the triage line clearly
- [ ] **Turn Wi-Fi off**, visibly

## The 3 minutes

| Time | Show | Say (one line) |
|---|---|---|
| 0:00 | Laptop, **Wi-Fi visibly off** | "No internet, no GPU, an ordinary laptop." |
| 0:15 | Upload a CNV sample | "Under a second: urgent referral, confidence, and where the model looked." |
| 0:45 | Switch to हिन्दी | "Same result in Hindi for operators who prefer it." |
| 1:05 | Upload a NORMAL sample | "Healthy scans clear quickly, so specialists see only who needs them." |
| 1:25 | Upload a selfie | "It refuses inputs that aren't OCT scans." |
| 1:40 | The borderline/low-confidence case | "When unsure, it says so and asks for human review." |
| 2:00 | About panel / benchmark slide | "Measured: X MB, Y ms on a 2015 i3, Z% macro-F1 on a patient-level test split." |
| 2:30 | Close | "Same intelligence, a fraction of the size: Advanced Frugal Innovation." |

## If something breaks (in this order)

1. Aditya's laptop (same `run_app.ps1`)
2. HF Space link → Render link
3. CLI in a terminal: `uv run python -m octscreen.app.cli --images demo --lang hi`
4. Backup video

## Judge questions: one-liners

- *Does it need the internet?* No. App and model run locally; we demo with Wi-Fi off. The link is just a convenience.
- *Wrong image?* A guard refuses colour photos, tiny images and odd shapes, and anything the model isn't confident about goes to human review instead of being guessed.
- *How did you pick the threshold?* From a validation-set table of flagged-rate vs silently missed urgent cases. The rule and value are documented.
- *Who is it for?* Technicians and optometrists at clinics with an OCT scanner but no on-site retina specialist, in English or Hindi.
- *Can it run elsewhere?* Same code on a laptop, in Docker, on Hugging Face and on Render. The model is a separate, versioned package with a checksum.

## Recording

Backup video **Sat** (OBS Studio, or Windows Game Bar `Win+Alt+R`), exactly this script. Final submission video **Sun**, Wi-Fi off on camera.
