@'
# Model Card

## 1. Model Details

### Final model

The final student architecture is **EfficientNet-Lite0**, trained using knowledge distillation from a ResNet-50 teacher.

| Parameter | Details |
|---|---|
| Final student architecture | EfficientNet-Lite0 |
| Teacher architecture | ResNet-50 |
| Deployment precision | INT8 |
| INT8 model size | 3.80 MB |
| Input resolution | 224 × 224 |
| Training framework | PyTorch |
| Inference runtime | ONNX Runtime |

The final model was selected after MobileNetV3-Small failed to maintain sufficient agreement with its FP32 version following INT8 quantisation.

The exact model package version, package metadata, and final run identifier will be added when the final model package is available.

### Intended use

The model is intended to provide screening and triage support for trained operators at clinics with access to OCT imaging equipment.

It is a research prototype and is not intended to replace clinical assessment or professional diagnosis.

### Out-of-scope use

The model is not intended for:

- Providing an independent medical diagnosis.
- Screening children.
- Processing non-OCT images.
- Use with scanners other than the training data source without appropriate validation.
- Making clinical decisions without human review.

## 2. Data

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

## 3. Training

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

### Model architectures and experiments

Four initial model runs were completed.

| Run | Architecture | Role |
|---|---|---|
| `teacher_r50_v1` | ResNet-50 (`resnet50.tv2_in1k`) | Teacher |
| `teacher_effb3_v1` | EfficientNet-B3 (`efficientnet_b3.ra2_in1k`) | Alternative teacher |
| `student_base_v1` | MobileNetV3-Small (`mobilenetv3_small_100.lamb.in1k`) | Baseline student |
| `student_kd_v1` | MobileNetV3-Small (`mobilenetv3_small_100.lamb.in1k`) | Initial knowledge-distilled student |

MobileNetV3-Small was evaluated as the initial student architecture. Its INT8 version achieved only 27–41% agreement with its FP32 version.

The student architecture was subsequently changed to EfficientNet-Lite0, which was retrained using the same ResNet-50 teacher.

**Final student architecture: EfficientNet-Lite0.**

The exact run identifier and resolved configuration filename for the final EfficientNet-Lite0 run are pending confirmation from the training artefacts.

### Knowledge distillation

The ResNet-50 teacher was used to train the final EfficientNet-Lite0 student through knowledge distillation.

The initial MobileNetV3-Small knowledge-distillation experiment used the following configuration:

| Parameter | Value |
|---|---:|
| Teacher architecture | ResNet-50 |
| Initial student architecture | MobileNetV3-Small |
| Temperature | 4.0 |
| Alpha | 0.7 |

The final EfficientNet-Lite0 run used the same teacher. Its exact distillation hyperparameters should be confirmed from its resolved training configuration before being reported.

### Training outputs

The following artefacts are retained for the initial model runs:

- `best.pt` — best-performing model checkpoint.
- `config_resolved.yaml` — resolved training configuration.
- `history.csv` — epoch-wise training history.
- `metrics_val.json` — validation metrics.
- `curves.png` — training and validation curves.

**Artefact location:** `OCT-Hackathon/model_checkpoints/` on Google Drive.

### Training duration

Exact wall-clock training durations were not recorded for the completed runs and are therefore not reported.

## 4. Evaluation

### Recorded results

The final EfficientNet-Lite0 knowledge-distilled student produced the following recorded results:

| Metric | Result |
|---|---:|
| Validation macro-F1 | 0.919 |
| Baseline validation macro-F1 | 0.888 |
| Macro-F1 improvement | 3.1 percentage points |
| INT8 model size | 3.80 MB |
| INT8-to-FP32 agreement | 99.5% |
| Agreement evaluation samples | 400 test scans |

The INT8 model achieved 99.5% agreement with the uncompressed model on 400 test scans.

The validation macro-F1 comparison is reported separately from the INT8-to-FP32 agreement result.

The final held-out test macro-F1, per-class metrics, AUC and measured inference latency are not included here because the corresponding results have not yet been provided.

## 5. Limitations

- The model was developed using the Kermany et al. OCT2017 dataset.
- The dataset represents a limited imaging domain and does not establish generalisation across different OCT scanners.
- The model has not been clinically validated.
- INT8-to-FP32 agreement measures consistency between model versions; it does not establish clinical accuracy.
- The reported validation macro-F1 is not a substitute for final held-out test performance.
- Input safety checks are heuristic checks and cannot guarantee that every accepted image is a valid OCT scan.
- Low-confidence predictions require human review.

## 6. Ethical Considerations

### Privacy

The application is designed to support offline inference so that image processing can take place locally.

The online demonstration is intended for sample scans only. Real patient data must not be uploaded to the public demonstration.

### Human oversight

The model provides screening support rather than an independent diagnosis. Cases with insufficient confidence are flagged for human review.

### Accessibility

The application supports English and Hindi to make the interface accessible to operators who prefer either language.

### Responsible use

The model should not be used as the sole basis for medical decisions. Its limitations, intended use and validation status should be communicated to operators.

## 7. How to Report Problems

Problems relating to the application, model predictions, input validation, deployment or user interface should be reported through the project repository.

When reporting an issue, include:

- A description of the problem.
- The application and model package version, if available.
- The steps required to reproduce the issue.
- The relevant error message or application log.

Do not attach real patient images or personally identifiable medical information to public issue reports.
'@ | Set-Content -Path docs/model_card.md -Encoding utf8