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