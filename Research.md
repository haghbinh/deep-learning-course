# Research Road Map: Explainable Fault Diagnosis via GAF/MTF and Grad-CAM

Oct 10, 2026 · @hh

## Overview

This research proposes an explainable deep learning pipeline for rotating machinery fault diagnosis. Raw vibration signals are converted into 2-D image representations — Gramian Angular Field (GAF) and Markov Transition Field (MTF) — and fed into a Convolutional Neural Network (CNN) for classification. Gradient-weighted Class Activation Mapping (Grad-CAM) then highlights the image regions that most influenced each classification decision, making the model's reasoning transparent and physically interpretable.

**Core research question:** Which time-frequency patterns encoded in GAF/MTF images does the CNN rely on to distinguish fault types, and do these patterns align with known fault signatures?

**Expected contributions:**

1. A validated signal-to-image encoding pipeline for fault diagnosis
2. A trained CNN classifier with competitive accuracy on benchmark datasets
3. Grad-CAM visualizations that link image activations back to physical fault mechanisms

## Phase 1 — Background & Literature Review

**Goal:** Build a solid theoretical foundation before any coding begins. Estimate 2–3 weeks.

**Topics to study:**

1. **Fault diagnosis fundamentals** — types of rotating machinery faults (bearing inner/outer race, ball fault, rotor imbalance), vibration signal characteristics, and traditional signal processing (FFT, envelope analysis, STFT).
2. **Time-series-to-image encoding** — understand GAF (Gramian Angular Summation Field, Gramian Angular Difference Field) and MTF mathematically; read the original Wang & Oates (2015) paper.
3. **CNNs for image classification** — convolutional layers, pooling, batch normalization, transfer learning with pretrained backbones (ResNet, VGG, EfficientNet).
4. **Explainability / XAI** — CAM, Grad-CAM (Selvaraju et al., 2017), Grad-CAM++, LIME, SHAP; understand what a saliency map communicates and its limitations.
5. **Related work** — survey at least 10–15 recent papers (2019–2025) on CNN-based fault diagnosis using image representations; note which datasets and metrics they use.

**Key papers to start with:**

- Wang & Oates (2015) — *Encoding Time Series as Images for Classification*
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*
- Wen et al. (2018) — *A New Convolutional Neural Network-Based Data-Driven Fault Diagnosis Method*
- Recent XAI-in-fault-diagnosis surveys (search IEEE Xplore, Scholar)

**Deliverable:** A written literature summary (2–3 pages) and an annotated bibliography.

## Phase 2 — Data Preparation & Signal-to-Image Encoding

**Goal:** Produce a clean, labeled image dataset ready for CNN training. Estimate 2 weeks.

**Steps:**

1. **Dataset acquisition** — use a public benchmark such as CWRU (Case Western Reserve University Bearing Dataset), MFPT, or PaderBorn. Download raw vibration signals for multiple fault types and severities.
2. **Pre-processing** — remove DC offset, apply a Hanning window, normalize amplitudes to \[−1, 1\]. Segment signals into fixed-length windows (e.g., 1024 or 4096 samples) with 50% overlap.
3. **GAF encoding** — (a) normalize each segment to \[−1, 1\] to obtain angular cosine values; (b) construct the Gramian Angular Summation Field (GASF) and/or Difference Field (GADF) matrix; (c) save as grayscale or color image.
4. **MTF encoding** — (a) quantize signal into Q bins; (b) build the Markov Transition Matrix; (c) assign probabilities along the temporal axis to form the MTF image.
5. **Multi-channel fusion (optional)** — stack GASF, GADF, and MTF as RGB channels for richer representation.
6. **Dataset split** — 70% train / 15% validation / 15% test; ensure class balance across fault types.

**Deliverable:** Saved image dataset (PNG or HDF5), class distribution table, and a short notebook demonstrating one encoding example per fault type.

## Phase 3 — CNN Model Design & Training

**Goal:** Train a classifier that reliably identifies fault types from GAF/MTF images. Estimate 2–3 weeks.

**Steps:**

1. **Baseline model** — design a compact custom CNN (3–5 conv blocks + fully connected head) to verify the pipeline end-to-end before applying larger models.
2. **Transfer learning** — fine-tune a pretrained backbone (e.g., ResNet-18 or EfficientNet-B0) by replacing the final classification head with a layer matching the number of fault classes.
3. **Training setup** — use cross-entropy loss, Adam optimizer (lr ≈ 1e-4), cosine LR scheduler, and early stopping on validation loss. Apply data augmentation (random flips, slight rotations, brightness jitter) if the dataset is small.
4. **Hyperparameter tuning** — experiment with learning rate, batch size, dropout rate, and number of frozen backbone layers. Keep a simple grid or random search log.
5. **Logging** — record training/validation loss and accuracy curves; save the best checkpoint.

**Key metrics:** Overall accuracy, per-class precision/recall/F1, confusion matrix.

**Deliverable:** Trained model checkpoint, training curves, and a confusion matrix for the test set.

## Phase 4 — Explainability with Grad-CAM

**Goal:** Produce interpretable heatmaps that reveal which regions of the GAF/MTF image drive each fault classification. Estimate 2 weeks.

**Steps:**

1. **Grad-CAM implementation** — hook into the final convolutional layer of the trained CNN; compute the gradient of the class score with respect to feature map activations; weight the maps by their global-average-pooled gradients and apply ReLU.
2. **Heatmap generation** — upsample the Grad-CAM output to the input image size; overlay it as a color-jet (or viridis) heatmap on the original GAF/MTF image.
3. **Per-class visualization** — generate heatmaps for every fault class and multiple test samples; examine whether the activated regions are consistent within a class.
4. **Back-mapping to the time domain** — the GAF/MTF encoding is invertible (approximately); map the highlighted image regions back to the original time-series windows to identify which temporal segments carry the fault signature.
5. **Qualitative analysis** — compare the Grad-CAM hotspots against known fault frequencies (BPFO, BPFI, BSF, FTF for bearings); a hotspot at an image location corresponding to a known fault harmonic strengthens interpretability.
6. **Optional extensions** — Grad-CAM++ for multi-scale activations; Score-CAM for gradient-free comparison; SHAP DeepExplainer for feature attribution.

**Deliverable:** A figure set of heatmap overlays per fault class, a written analysis linking hotspots to physical fault mechanisms, and the back-mapping notebook.

## Phase 5 — Evaluation & Validation

**Goal:** Rigorously benchmark the full pipeline and demonstrate both diagnostic accuracy and explanation quality. Estimate 1–2 weeks.

**Quantitative evaluation:**

| Aspect | Metric | Target |
| --- | --- | --- |
| Classification | Accuracy, F1 (macro) | ≥ 95% on CWRU |
| Robustness | Accuracy under added Gaussian noise (SNR 0–10 dB) | Graceful degradation |
| Explanation faithfulness | Deletion / Insertion AUC (pixel-flipping test) | Higher than random |
| Explanation consistency | Intersection-over-Union of Grad-CAM masks across same-class samples | ≥ 0.5 |

**Qualitative evaluation:**

- Show that Grad-CAM highlights match fault-frequency regions identified by an expert or by classical signal analysis.
- Conduct a brief user study or expert review if feasible.

**Ablation studies:**

- Compare GASF-only vs. GADF-only vs. MTF-only vs. multi-channel fusion.
- Compare Grad-CAM vs. Grad-CAM++ vs. Score-CAM.
- Compare custom CNN vs. fine-tuned ResNet.

**Deliverable:** Results tables, ablation comparison, and a clear conclusion on which configuration best balances accuracy and interpretability.

## Recommended Tools, Libraries & Datasets

**Python libraries:**

| Library | Purpose |
| --- | --- |
| `pyts` | GAF and MTF encoding (built-in `GramianAngularField`, `MarkovTransitionField`) |
| `PyTorch` + `torchvision` | CNN training and pretrained backbones |
| `pytorch-grad-cam` | Grad-CAM, Grad-CAM++, Score-CAM implementations |
| `scikit-learn` | Metrics, confusion matrix, stratified splits |
| `matplotlib` / `seaborn` | Visualization of heatmaps and training curves |
| `numpy` / `scipy` | Signal pre-processing and windowing |

**Benchmark datasets:**

| Dataset | Fault types | Notes |
| --- | --- | --- |
| CWRU Bearing | Inner race, outer race, ball, normal | Most widely used; 12 kHz and 48 kHz |
| MFPT | Inner race, outer race | Real machine data |
| PaderBorn | Multiple bearing damage modes | Artificial + real damage |
| PRONOSTIA (FEMTO) | Bearing run-to-failure | Good for degradation studies |

**Development environment:** Python 3.10+, CUDA-enabled GPU (optional but recommended for training), Jupyter notebooks for exploration, then modular `.py` scripts for the final pipeline.
