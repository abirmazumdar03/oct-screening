# Model Card

## 1. Data

### Dataset source

The models were trained using the Kermany et al. (2018) OCT2017 dataset, obtained from the Kaggle `kermany2018` collection.

**Dataset citation:** Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.

**Dataset usage:** The dataset is intended for research use. Refer to the original dataset terms when redistributing or using the images.

### Dataset composition

The dataset contains four categories:

- CNV (Choroidal Neovascularisation)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

After duplicate filename removal, the final dataset contained:

| Description | Count |
|---|---:|
| Total images | 83,605 |
| Total patients | 4,657 |
| Duplicate filenames skipped | 879 |

### Data splitting

A patient-level stratified group splitting strategy was used to prevent data leakage.

| Split | Patients | Images |
|---|---:|---:|
| Training | 3,735 | 67,807 |
| Validation | 460 | 7,980 |
| Testing | 462 | 7,818 |
| **Total** | **4,657** | **83,605** |

- Target split proportions: 80% training, 10% validation and 10% testing.
- Random seed: 42.
- All scans belonging to a patient were assigned to exactly one partition.

### Preprocessing

All input OCT images underwent the following preprocessing:

1. Conversion to grayscale.
2. Resizing to 224 × 224 pixels using squash resizing.
3. Conversion to three channels.
4. ImageNet normalisation.

Training-time augmentation included horizontal flipping, rotation, translation, scaling, brightness adjustment and contrast adjustment.

## 2. Training

### Training environment

| Parameter | Configuration |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA T4 |
| Framework | PyTorch |
| Model library | timm |
| Precision | Automatic Mixed Precision (AMP) |
| Input resolution | 224 × 224 |
| Random seed | 42 |

### Model architectures

Four model runs were completed.

| Run | Architecture | Role |
|---|---|---|
| `teacher_r50_v1` | ResNet-50 (`resnet50.tv2_in1k`) | Teacher |
| `teacher_effb3_v1` | EfficientNet-B3 (`efficientnet_b3.ra2_in1k`) | Alternative teacher |
| `student_base_v1` | MobileNetV3-Small (`mobilenetv3_small_100.lamb_in1k`) | Baseline student |
| `student_kd_v1` | MobileNetV3-Small (`mobilenetv3_small_100.lamb_in1k`) | Knowledge-distilled student |

### Training configurations

The resolved configurations are retained with their corresponding run artefacts.

| Run | Configuration file |
|---|---|
| `teacher_r50_v1` | `teacher_r50_v1/config_resolved.yaml` |
| `teacher_effb3_v1` | `teacher_effb3_v1/config_resolved.yaml` |
| `student_base_v1` | `student_base_v1/config_resolved.yaml` |
| `student_kd_v1` | `student_kd_v1/config_resolved.yaml` |

### Knowledge distillation

The ResNet-50 teacher was used to train the MobileNetV3-Small student through knowledge distillation.

| Parameter | Value |
|---|---:|
| Teacher architecture | ResNet-50 |
| Student architecture | MobileNetV3-Small |
| Temperature | 4.0 |
| Alpha | 0.7 |

The student was trained both independently and with teacher guidance to measure the effect of knowledge distillation.

### Training outputs

The following artefacts are retained for each run:

- `best.pt` — best-performing model checkpoint.
- `config_resolved.yaml` — resolved training configuration.
- `history.csv` — epoch-wise training history.
- `metrics_val.json` — validation metrics.
- `curves.png` — training and validation curves.

**Artefact location:** `OCT-Hackathon/model_checkpoints/` on Google Drive.

### Training duration

Exact wall-clock training durations were not recorded for the completed runs and are therefore not reported.

## 3. Model details (Poddar, from model_meta.json)

| | |
|---|---|
| Package | _from `model_meta.json`: package_name vX.Y.Z_ |
| Task | 4-class classification of retinal OCT B-scans: CNV, DME, DRUSEN, NORMAL |
| Architecture | _student architecture (Aditya)_, distilled from _teacher (Aditya)_ |
| Format / precision | ONNX, _INT8 / FP32 (Abir)_; Grad-CAM heatmaps from the FP32 twin (`explainer.pt`, optional) |
| Size | _X MB (Abir)_ |
| Runtime | ONNX Runtime, CPU only, no internet |
| Calibration | temperature _T (Abir)_ applied to the logits before softmax |
| Version / date | _vX.Y.Z, Sat 7 Nov 2026_ |
| Contact | the team (see README) |

## 4. Intended use (Poddar)

**Primary use:** screening *triage support* for trained operators (technicians, optometrists) at clinics that have an OCT scanner but no on-site retina specialist. The tool helps decide **who to refer first**: 🔴 urgent referral (CNV, DME), 🟡 routine follow-up (Drusen), 🟢 no signs of disease, ⚪ uncertain → manual review.

**Users:** operators with no AI knowledge, in English or Hindi. Every result carries the disclaimer that the final decision rests with a qualified clinician.

**Out of scope:**
- diagnosis or treatment decisions without a clinician
- children (no paediatric data in training)
- images that are not retinal OCT B-scans (fundus photos, other modalities, screenshots); the input guard catches many, but not all
- OCT scanners other than the training source **without local validation first**
- conditions outside the 4 classes (e.g. glaucoma, retinal detachment): the model will still output one of its 4 classes

## 5. Evaluation (Abir)

_Patient-level test split: accuracy, macro-F1, per-class sensitivity/specificity, urgent sensitivity (CNV + DME), confusion matrix, calibration (ECE before/after temperature), latency on the 3 laptops (median, p95)._

**Confidence threshold:** _value_, chosen on the **validation** split with the rule "lowest threshold with silent urgent misses ≤ 0.5 % while flagged-uncertain ≤ 10 %". Result: _x % flagged, y % silent urgent misses_ (see `docs/decisions.md`).

## 6. Limitations (Abir)

_Single-dataset, single-scanner-brand evaluation; no prospective/clinical validation; Grad-CAM shows where the FP32 twin looked, which is not a proof of reasoning; guard is heuristic (grayscale non-OCT images can pass)._

## 7. Ethical considerations (Poddar)

- **Privacy by design:** the intended deployment is an offline laptop. Scans are processed in memory and never leave the machine; the app's temporary upload folder is wiped on every start (`run_app.ps1`). Gradio telemetry is disabled. The online demo is for judges only and asks users not to upload patient data.
- **Uncertainty is shown, not hidden:** below the confidence threshold the app never shows a disease verdict on its own; it says ⚪ *refer for manual review* and explains why.
- **Refuses what it shouldn't judge:** colour photos, tiny images and implausible shapes are rejected before the model runs.
- **Human in the loop:** every screen carries "Screening aid only, not a diagnosis". The triage map is clinic policy, kept in a config file (`configs/app.yaml`), not hidden in code.
- **Accessible:** every visible string exists in English and Hindi; triage uses an emoji **and** words, never colour alone. Medical wording in Hindi was reviewed by _name / role_.
- **Failure cost asymmetry:** a missed urgent case (CNV/DME) is worse than a false alarm; the threshold rule optimises for few silent urgent misses.

## 8. How to report problems

Open an issue in the GitHub repo with: the app version (About panel → model package), the language, what you uploaded (**never a real patient scan**; describe it instead), what you expected and what happened. For a wrongly rejected real OCT scan, include the "colour score" or size shown in the message.
