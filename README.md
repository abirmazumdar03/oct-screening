# OCT Retina Screening Assistant (ZEISS BeyondCode'26, AFI theme)

Small, offline AI for screening retinal OCT scans on an ordinary laptop. **Screening aid only, not a diagnosis.**

## Setup (Windows, PowerShell)
```powershell
cd C:\dev ; git clone https://github.com/<you>/oct-screening.git ; cd oct-screening
uv sync --group train        # Aditya  |  --group optimize (Abir)  |  --group app --group app-heatmap (Poddar)
uv run pytest -q
```
Team guides: see the work-split docs (00 foundation, 01 Aditya, 02 Abir, 03 Poddar). Full README comes at the end of the sprint.
