# Training Report

## 1. Data

- **Source:** Kermany et al. (2018), OCT2017 dataset, obtained from the Kaggle `kermany2018` collection.
- **Original classes:** CNV, DME, DRUSEN and NORMAL.
- **Duplicate handling:** 879 duplicate filenames were skipped during dataset discovery.
- **Final dataset:** 83,605 images from 4,657 patients.

### Patient-level split

The dataset was divided at the patient level to prevent scans belonging to the same patient from appearing in different splits.

- **Method:** Patient-level stratified group splitting.
- **Target proportions:** 80% training, 10% validation and 10% testing.
- **Random seed:** 42.
- **Split fingerprint:** To be added from the split metadata.

| Split | Patients | Images | Proportion |
|---|---:|---:|---:|
| Training | 3,735 | 67,807 | 81.1% |
| Validation | 460 | 7,980 | 9.5% |
| Testing | 462 | 7,818 | 9.4% |
| **Total** | **4,657** | **83,605** | **100%** |

### Class distribution

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| CNV | 29,408 | 4,053 | 3,755 |
| DME | 9,722 | 688 | 1,012 |
| DRUSEN | 7,116 | 792 | 712 |
| NORMAL | 21,561 | 2,447 | 2,339 |
| **Total** | **67,807** | **7,980** | **7,818** |

### Preprocessing

The OCT images were processed using the following pipeline:

1. Convert the input image to grayscale.
2. Resize to 224 × 224 pixels using the squash resize strategy.
3. Convert the grayscale image into three channels.
4. Apply ImageNet normalisation.

### Data augmentation

The following augmentations were applied during training:

| Augmentation | Configuration |
|---|---|
| Horizontal flip | Probability 0.5 |
| Rotation | ±10° |
| Translation | 5% |
| Scaling | 0.95–1.05 |
| Brightness | ±10% |
| Contrast | ±10% |

## 2. Models and Runs

Four experiments were conducted: two teacher models, a student trained independently and a student trained using knowledge distillation.

All experiments were performed on a Kaggle GPU environment using an NVIDIA T4 GPU with mixed-precision training.

| Run | Architecture | Best Epoch | Validation Accuracy | Validation Macro-F1 | Urgent Sensitivity |
|---|---|---:|---:|---:|---:|
| teacher_r50_v1 | ResNet-50 | 9 | 0.9526 | 0.9238 | 0.9755 |
| teacher_effb3_v1 | EfficientNet-B3 | 3 | 0.9291 | 0.8906 | 0.9589 |
| student_base_v1 | MobileNetV3-Small | 20 | 0.9229 | 0.8799 | 0.9700 |
| student_kd_v1 | MobileNetV3-Small | 10 | 0.9291 | 0.8868 | 0.9673 |

*Note: Epoch numbers are reported using one-based counting. Training duration and exact parameter counts should be added from the corresponding run artefacts if required.*

### Teacher model comparison

ResNet-50 achieved the highest validation macro-F1, accuracy and urgent sensitivity among the two teacher models. It was therefore retained as the selected teacher for knowledge distillation.

EfficientNet-B3 did not outperform ResNet-50 on any of the three reported validation metrics.

## 3. Knowledge Distillation

Knowledge distillation was performed using the ResNet-50 teacher and MobileNetV3-Small student.

The configuration used was:

| Parameter | Value |
|---|---:|
| Teacher | ResNet-50 |
| Student | MobileNetV3-Small |
| Temperature (T) | 4.0 |
| Alpha (α) | 0.7 |

### Baseline versus KD student

| Metric | Baseline Student | KD Student | Change |
|---|---:|---:|---:|
| Validation Accuracy | 0.9229 | 0.9291 | +0.61 percentage points |
| Validation Macro-F1 | 0.8799 | 0.8868 | +0.69 percentage points |
| Urgent Sensitivity | 0.9700 | 0.9673 | -0.27 percentage points |

Knowledge distillation improved the student's validation accuracy and macro-F1 compared with independent training. However, urgent sensitivity decreased slightly.

The KD student achieved a macro-F1 of 0.8868, while the ResNet-50 teacher achieved 0.9238. The resulting gap is approximately 3.70 percentage points. Therefore, the student did not meet the target of remaining within 3 percentage points of the teacher.

No additional temperature or alpha sweep was conducted.

Based on validation macro-F1, the KD student is the leading student candidate. The baseline is retained as a comparison model.

## 4. Observations

- ResNet-50 provided the strongest validation performance among the evaluated models.
- EfficientNet-B3 did not provide an improvement over the selected ResNet-50 teacher.
- Knowledge distillation produced a measurable improvement in student macro-F1 and accuracy.
- The KD student achieved higher macro-F1 than the independently trained student, but its urgent sensitivity was slightly lower.
- The KD student remains approximately 3.70 percentage points behind the teacher in macro-F1.
- Patient-level splitting was used to prevent patient overlap across the dataset partitions.

*The per-class confusion patterns and specific misclassification counts have not been included here because the corresponding validation confusion-matrix results have not yet been added to this report.*

## 5. Limitations

1. **Single public dataset:** The experiments were conducted using the Kermany et al. OCT2017 dataset. Performance on independent datasets remains unverified.
2. **Scanner variation:** The dataset primarily represents scans from a limited scanner domain. Generalisation to other OCT scanner manufacturers, including ZEISS devices, has not been established.
3. **No external validation:** The models have not yet been evaluated on an independent external clinical dataset.
4. **Student-teacher performance gap:** The KD student improved over the baseline but did not reach the target of being within 3 percentage points of the teacher's validation macro-F1.
5. **Deployment evaluation pending:** Model compression, INT8 quantisation, compressed model size and inference latency require separate measurement.
6. **Clinical scope:** The system is a research prototype intended as a screening aid, not a diagnostic tool or clinically validated medical device.

## 6. Artefacts and Reproducibility

The following run artefacts have been backed up to Google Drive:

- `best.pt`
- `config_resolved.yaml`
- `history.csv`
- `metrics_val.json`
- `curves.png`

The artefacts are available for the following runs:

- `teacher_r50_v1`
- `teacher_effb3_v1`
- `student_base_v1`
- `student_kd_v1`

The training configurations and validation metrics are retained alongside the model checkpoints to support reproducibility and subsequent deployment evaluation.

**Important:** All performance values in this report are validation results. The held-out test split has not been used for model selection.