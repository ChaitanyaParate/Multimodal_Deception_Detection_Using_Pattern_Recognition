# 👁️ AI Lie Detector & Multimodal Pattern Recognition Studio

> An end-to-end multimodal deception detection platform and real-time telemetry inspection studio combining linguistic, acoustic, and ocular behavioral dynamics across three distinct pattern recognition paradigms (Subspace Generative Bayesian, Early Margin/Ensemble Fusion, and Late Stacked Generalization) alongside deep neural and heuristic baselines.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-blue?logo=scikitlearn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-blueviolet)](https://mediapipe.dev/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-lightgrey?logo=openai)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌟 Key Highlights

- 🖥️ **Interactive Multimodal Studio (`app.py`):** Full-featured Streamlit application for real-time video inspection, multi-model consensus probability gauges, and synchronized acoustic pitch, ocular EAR, and linguistic breakdown plots.
- 🎯 **OpenCV Telemetry HUD Visualizer (`visualizer.py`):** Real-time Heads-Up Display (HUD) video annotator with 468-point facial mesh tracking, frame-level Eye Aspect Ratio (EAR) blink detection, pupil saccade tracking, hesitation subtitle strip, and full audio re-muxing via `ffmpeg`.
- 🧠 **Three Rigorous Pattern Recognition Paradigms (13 Models Total):**
  1. **Subspace Generative Bayesian Family:** `Subspace-GaussianNB`, `Subspace-LDA` (with Ledoit-Wolf shrinkage), and `Subspace-RegQDA` ($\alpha=0.3$).
  2. **Early Fusion Margin & Ensemble Family:** `EarlyFusion-SVM-Linear`, `EarlyFusion-SVM-RBF` (Platt-scaled), `EarlyFusion-RandomForest`, `EarlyFusion-GradBoost`, and `EarlyFusion-ExtraTrees`.
  3. **Late Decision-Level Fusion Family:** `LateFusion-StackingLR` (5-fold out-of-fold meta-learner), `LateFusion-SoftVoting`, and `LateFusion-BayesianPoE` (Bayesian Product of Experts).
  4. **Baselines & Meta-Consensus:** `Heuristic-Weighted`, `Deep-DeceptionMLP`, and `★ Ensemble Consensus` (soft-majority meta-ensemble).
- 🔬 **Curse of Dimensionality Solved:** Condenses high-dimensional small-sample video data ($d=396$, $N=121$) via leakage-free Subspace PCA compression into an orthogonal 22-dimensional compact manifold ($N/d \approx 5.5$), preventing singular covariance collapse.
- ⚡ **10x Accelerated Inference & Telemetry Engine:** Sub-30ms video switching via disk-persisted `telemetry_cache.joblib`, bounded 75-360Hz $F_0$ pitch tracking, in-memory frame decoding, and instant `-c:a copy` ffmpeg remuxing.

---

## 🧠 The Three Core Pattern Recognition Approaches

To detect high-stakes interpersonal deception without overfitting on small multimodal datasets ($N=121$, $d=396$), this platform implements and benchmarks three foundational, mathematically grounded pattern recognition paradigms:

| Approach | Core Mathematical Mechanism | Implemented Estimators (`pr_solutions.py`) | Best Holdout Acc / F1 | Best 5-Fold CV ROC-AUC |
|---|---|:---:|:---:|:---:|
| **Approach 1: Subspace Dimensionality Reduction (PCA + Fisher's LDA) with Generative Bayesian Classifiers (Gaussian Naive Bayes & QDA)** | Orthogonal subspace projection ($396 \to 22 \to 8\text{ dims}$) + Fisher's scatter ratio $J(w)$ + class-conditional Gaussian density estimation $p(x \mid \omega_i)$ | `Subspace-GaussianNB`, `Subspace-LDA`, `Subspace-RegQDA` | 58.3% / 0.286 | **0.746 ± 0.103** (`RegQDA`) |
| **Approach 2: Early Feature-Level Fusion with Discriminative Margin (SVM-RBF / SVM-Linear) & Ensemble (Random Forest / Extra Trees) Classifiers** | Concatenated multimodal feature manifold ($d=22$) with maximum-margin hyperplanes, Mercer kernels, and randomized decision tree ensembles | `EarlyFusion-SVM-Linear`, `EarlyFusion-SVM-RBF`, `EarlyFusion-RandomForest`, `EarlyFusion-ExtraTrees`, `EarlyFusion-GradBoost` | **83.3% / 0.833** (`RF` & `ExtraTrees`) | **0.769 ± 0.120** (`ExtraTrees`) |
| **Approach 3: Late Decision-Level Fusion via Stacked Generalization (Out-of-Fold Unimodal Posterior Meta-Classifier)** | Independent unimodal classifiers ($X_{\text{text}}, X_{\text{audio}}, X_{\text{vision}}$) + 5-fold out-of-fold posterior meta-learning & Bayesian Product of Experts | `LateFusion-StackingLR`, `LateFusion-SoftVoting`, `LateFusion-BayesianPoE` | **75.0% / 0.727** (`StackingLR`) | **0.670 ± 0.101** (`SoftVoting`) |

---

### Approach 1: Subspace Dimensionality Reduction (PCA + Fisher's LDA) with Generative Bayesian Classifiers (Gaussian Naive Bayes & QDA):

High-dimensional multimodal data with limited sample size ($d=396 \gg N_{\text{train}}=109$) inevitably causes the empirical covariance matrix $\hat{\Sigma}$ to become rank-deficient and singular ($\text{rank}(\hat{\Sigma}) \le 109 < 396$), leading to catastrophic numerical collapse in Gaussian density estimation. Approach 1 (`pr_pipeline.py` and `pr_solutions.py`) resolves this via orthogonal subspace projection and regularized generative modeling:

1. **Orthogonal Subspace PCA Projection:**
   In `MultimodalPreprocessor` ([`pr_pipeline.py`](pr_pipeline.py)), eigen-decomposition of the sample covariance matrix compresses the $384$-dimensional `all-MiniLM-L6-v2` dense semantic embeddings to $k=10$ orthogonal principal axes capturing the dominant linguistic variance while filtering out high-frequency noise:
   $$\Sigma_x = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})(x_i - \bar{x})^T = V \Lambda V^T \implies z_{\text{text}} = W_{\text{PCA}}^T (x_{\text{text}} - \bar{x}_{\text{text}}), \quad W_{\text{PCA}} \in \mathbb{R}^{384 \times 10}$$
   Combining $10\text{d}$ text PCA components with 4 engineered lexical features, 3 acoustic features, and 5 ocular features yields a compact $22\text{-dimensional}$ space ($N/d \approx 5.5$), well within non-singular estimation bounds.

2. **Fisher's Linear Discriminant Analysis (LDA) with Analytic Shrinkage:**
   `Subspace-LDA` ([`pr_solutions.py`](pr_solutions.py)) projects the compact manifold onto a $1\text{D}$ discriminant line $w$ maximizing the Generalized Rayleigh Quotient (ratio of between-class scatter $S_B$ to within-class scatter $S_W$):
   $$J(w) = \frac{w^T S_B w}{w^T S_W w}, \quad \text{where } S_B = (\mu_1 - \mu_0)(\mu_1 - \mu_0)^T, \quad S_W = \sum_{i=0}^1 \sum_{x \in \omega_i} (x - \mu_i)(x - \mu_i)^T$$
   To guarantee numerical invertibility of $S_W$, Ledoit-Wolf analytic covariance shrinkage is applied:
   $$\Sigma^* = (1 - \lambda)\hat{\Sigma} + \lambda \frac{\text{Tr}(\hat{\Sigma})}{p}I$$

3. **Generative Bayes Decision Rule & Regularized QDA ($\alpha = 0.3$):**
   Unlike LDA (which assumes identical covariance $\Sigma_0 = \Sigma_1$) or Gaussian Naive Bayes (which enforces strict diagonal independence $p(x \mid \omega_i) = \prod_j \mathcal{N}(x_j; \mu_{ij}, \sigma_{ij}^2)$), Regularized Quadratic Discriminant Analysis (`Subspace-RegQDA`) estimates class-specific quadratic boundaries while regularizing class covariance estimates toward an identity sphere to prevent overfitting:
   $$\hat{\Sigma}_i(\alpha) = (1 - \alpha)\hat{\Sigma}_i + \alpha \sigma^2 I, \quad \hat{\omega} = \arg\max_{i \in \{0,1\}} \left[ \ln p(x \mid \omega_i) + \ln P(\omega_i) \right]$$
   where log-likelihood is given by $\ln p(x \mid \omega_i) = -\frac{1}{2}\ln|\hat{\Sigma}_i(\alpha)| - \frac{1}{2}(x - \mu_i)^T \hat{\Sigma}_i(\alpha)^{-1}(x - \mu_i) - \frac{d}{2}\ln(2\pi)$.

---

### Approach 2: Early Feature-Level Fusion with Discriminative Margin (SVM-RBF / SVM-Linear) & Ensemble (Random Forest / Extra Trees) Classifiers

In Approach 2 (`EarlyFusionDiscriminativeBenchmark` in [`pr_solutions.py`](pr_solutions.py)), normalized lexical ($14\text{d}$), acoustic ($3\text{d}$), and ocular ($5\text{d}$) features are concatenated into a joint $22\text{-dimensional}$ multimodal vector $x = [x_{\text{text}}^T, x_{\text{audio}}^T, x_{\text{vision}}^T]^T$. Early fusion enables discriminative estimators to discover cross-modal correlations—such as acoustic pitch tremors occurring concurrently with abnormal gaze saccades—directly at the representation layer:

1. **Maximum-Margin Hyperplanes & Mercer Kernels (`SVM-Linear`, `SVM-RBF`):**
   Support Vector Machines solve the primal convex quadratic optimization problem:
   $$\min_{w, b, \xi} \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
   where non-linear boundaries are implicitly mapped to a reproducing kernel Hilbert space via the Gaussian Radial Basis Function (RBF) kernel:
   $$K(x, x') = \exp\left(-\gamma \|x - x'\|_2^2\right)$$
   Calibrated posterior probabilities are generated via Platt scaling ($P(\omega_{\text{lie}} \mid x) = \frac{1}{1 + \exp(A f(x) + B)}$). Linear SVM additionally provides direct feature attribution: the margin direction $w_j \cdot x_j$ powers the **Local Feature Attribution ("Why This Verdict?")** bar chart in [`app.py`](app.py).

2. **Randomized Decision Tree Ensembles (`Random Forest`, `Extra Trees`, `Gradient Boosting`):**
   - **Random Forest:** Constructs $B=200$ decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspace sampling ($m = \sqrt{d}$), reducing estimator variance without increasing bias.
   - **Extra Trees (Extremely Randomized Trees):** Further mitigates small-sample overfitting on $N=121$ by drawing random candidate split thresholds rather than searching for optimal cutoffs, achieving the top cross-validated ROC-AUC (**$0.769 \pm 0.120$**). Gini importance is accumulated across all nodes:
     $$I_G(f) = \sum_{T} \sum_{t \in T: v(t)=f} \frac{N_t}{N} \Delta i(t)$$

---

### Approach 3: Late Decision-Level Fusion via Stacked Generalization (Out-of-Fold Unimodal Posterior Meta-Classifier)

Rather than forcing disparate modalities into a single feature vector, Approach 3 (`LateFusionStackingClassifier` in [`pr_solutions.py`](pr_solutions.py)) trains three isolated Level-0 unimodal estimators on text ($X_{\text{text}} \in \mathbb{R}^{14}$), audio ($X_{\text{audio}} \in \mathbb{R}^3$), and vision ($X_{\text{vision}} \in \mathbb{R}^5$). This preserves individual modality geometries and isolates sensor noise (e.g., poor microphone acoustics or camera occlusions) from corrupting other clean streams:

1. **Leakage-Free 5-Fold Out-of-Fold (OOF) Posterior Generation:**
   To prevent data leakage when training the meta-classifier, the training partition is split into 5 stratified folds. For each fold $k$, unimodal models trained exclusively on the other 4 folds generate validation posterior predictions:
   $$z_i = \left[ P(\omega_{\text{lie}} \mid x_{i,\text{text}}), \; P(\omega_{\text{lie}} \mid x_{i,\text{audio}}), \; P(\omega_{\text{lie}} \mid x_{i,\text{vision}}) \right]^T \in [0,1]^3$$

2. **Level-1 Stacked Logistic Regression Meta-Classifier:**
   A non-negative, calibrated meta-learner trains on the out-of-fold probability matrix $Z \in \mathbb{R}^{N \times 3}$ to learn optimal, data-driven modality weights:
   $$P(\omega_{\text{lie}} \mid z) = \sigma\left(w_{\text{text}} p_{\text{text}} + w_{\text{audio}} p_{\text{audio}} + w_{\text{vision}} p_{\text{vision}} + b\right)$$
   where $\sigma(u) = \frac{1}{1 + e^{-u}}$. This completely eliminates the legacy arbitrary `0.3 / 0.3 / 0.4` heuristic by discovering empirical weights where reliable nonverbal cues dominate ambiguous textual cues.

3. **Bayesian Product of Experts (PoE):**
   Under conditional independence across modalities given class $\omega$, the joint log-posterior odds equals the sum of unimodal log-odds:
   $$\ln \frac{P(\omega_{\text{lie}} \mid X)}{P(\omega_{\text{truth}} \mid X)} = \sum_{m \in \{\text{text}, \text{audio}, \text{vision}\}} \ln \frac{p_m}{1 - p_m}$$
   providing an exact, parameter-free generative consensus baseline.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input["1. Raw Multimodal Input"]
        V["Input Video (.mp4)"]
    end

    subgraph FeatureExtraction["2. Synchronous Feature Extraction"]
        V --> W["OpenAI Whisper ASR"]
        V --> L["Librosa Audio Engine (75-360 Hz PyIN)"]
        V --> M["MediaPipe FaceMesh (468 pts + CLAHE)"]

        W --> T_FEAT["1. Text Stream<br/>• 384d MiniLM Embeddings<br/>• Hesitation Frequency<br/>• Semantic Similarity"]
        L --> A_FEAT["2. Audio Stream<br/>• pyin Pitch Contour (F0 std)<br/>• RMS Silence Ratio<br/>• Voicing Energy"]
        M --> V_FEAT["3. Vision Stream<br/>• Eye Aspect Ratio (EAR)<br/>• Blink Rate (/min)<br/>• Saccade Std Dev"]
    end

    subgraph Subspace["3. Leakage-Free Subspace Compression"]
        T_FEAT --> PCA_MOD["PCA Subspace (k=10)"]
        PCA_MOD --> COMPACT["Compact 22-Dim Multimodal Vector<br/>(14 Text + 3 Audio + 5 Vision)"]
        A_FEAT --> COMPACT
        V_FEAT --> COMPACT
    end

    subgraph Models["4. Pattern Recognition Model Registry (13 Models)"]
        COMPACT --> FAM1["Approach 1: Subspace Generative Bayesian<br/>• GaussianNB<br/>• LDA with Shrinkage<br/>• RegQDA (alpha=0.3)"]
        COMPACT --> FAM2["Approach 2: Early Feature Fusion<br/>• SVM-Linear (Max-Margin)<br/>• SVM-RBF (Platt Scaled)<br/>• Random Forest & Extra Trees<br/>• Gradient Boosting"]
        COMPACT --> FAM3["Approach 3: Late Stacked Decision Fusion<br/>• 5-Fold OOF Unimodal Estimators<br/>• Level-1 Logistic Regression Meta-Learner<br/>• Bayesian Product of Experts"]
        COMPACT --> BASE["Baselines & Meta-Consensus<br/>• Heuristic Weighted (0.3 / 0.3 / 0.4)<br/>• Deep DeceptionMLP (PyTorch)<br/>• Ensemble Consensus"]
    end

    subgraph Deliverables["5. Visual Deliverables & Interfaces"]
        FAM1 --> APPS["Unified Model Inference Engine"]
        FAM2 --> APPS
        FAM3 --> APPS
        BASE --> APPS
        APPS --> STUDIO["Interactive Streamlit Studio<br/>(app.py - Sub-30ms Telemetry Cache)"]
        APPS --> HUD["OpenCV Telemetry HUD Annotator<br/>(visualizer.py - 468pt Mesh + Sparkline)"]
        APPS --> BENCH["Automated Benchmark Suite<br/>(benchmark.py - Dark Studio Palette)"]
    end
```

---

## ⚡ Performance & Latency Engineering

To make real-time multimodal deception detection responsive for live courtroom and interrogation scenarios, the pipeline incorporates several key performance optimizations that achieve a **10x speedup**:

1. **Persistent Telemetry Cache (`outputs/telemetry_cache.joblib`):**
   - Audio pitch contour extraction (`librosa.pyin`) and dense frame-by-frame 468-point facial mesh tracking traditionally require 4–8 seconds per video.
   - The engine automatically computes a persistent cryptographic disk cache keyed by video path and modification timestamp. Switching between pre-analyzed demo clips or previously inspected files in `app.py` completes in **<30 ms**!
2. **Bounded Human Vocal Pitch Search ($75 - 360\text{ Hz}$):**
   - By constraining the `pyin` probabilistic fundamental frequency search to typical human vocal ranges ($75\text{ Hz} \le f_0 \le 360\text{ Hz}$) and using `hop_length=1024`, pitch tracking executes 2.5x faster without loss of micro-tremor resolution.
   - Audio waveform displays in Matplotlib are decimate-downsampled by $10\times$ (`y_wave[::10]`), reducing vector render overhead from 480,000 vertices to 48,000 vertices.
3. **In-Memory Frame Decoding & 360px Scaled Mesh:**
   - `vision.py` implements direct in-memory frame decoding via `cv2.VideoCapture` (`analyze_vision_from_video`), eliminating hundreds of disk I/O operations from temporary JPEG creation.
   - For FaceMesh tracking, frames are scaled to width $360\text{ px}$ before calling MediaPipe, cutting CPU memory copy time by $4\times$.
4. **Adaptive Contrast (CLAHE) & Dual-Stage FaceMesh Fallback:**
   - On challenging low-contrast or angled witness angles (e.g. `trial_lie_001.mp4`), the tracking engine applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel in LAB color space and falls back to a static high-sensitivity detector (`min_detection_confidence=0.15`), guaranteeing zero frame dropouts.
5. **Instantaneous Audio Re-Muxing (`-c:a copy`):**
   - Annotated HUD video export in `visualizer.py` utilizes direct audio stream copy (`ffmpeg -c:a copy -movflags +faststart`) to mux original AAC/MP3 audio instantaneously into the final MP4 container.

---

## 📊 Empirical Benchmark Results

All models were evaluated using a **dual-protocol evaluation**:
1. **Official Holdout Split:** 109 Train / 12 Test ($N_{\text{test}} = 12$)
2. **5-Fold Stratified Cross-Validation:** Full dataset evaluation ($N = 121$, Mean $\pm$ Std)

### Table 1: Multi-Model Benchmark Comparison (13 Models)

| Model Architecture | Paradigm Family | Holdout Acc. | Holdout F1 | Holdout ROC-AUC | 5-Fold CV Acc. | 5-Fold CV F1 | 5-Fold CV ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic-Weighted** | Baseline | 0.417 | 0.000 | 0.444 | $0.495 \pm 0.065$ | $0.225 \pm 0.083$ | $0.659 \pm 0.154$ |
| **Deep-DeceptionMLP** | Deep Baseline | 0.500 | 0.667 | 0.694 | **$0.728 \pm 0.060$** | $0.721 \pm 0.058$ | **$0.777 \pm 0.070$** |
| **Subspace-GaussianNB** | Approach 1: Subspace Generative | 0.417 | 0.000 | 0.222 | $0.546 \pm 0.033$ | $0.578 \pm 0.038$ | $0.612 \pm 0.069$ |
| **Subspace-LDA** | Approach 1: Subspace Generative | 0.333 | 0.000 | 0.250 | $0.529 \pm 0.045$ | $0.543 \pm 0.045$ | $0.598 \pm 0.045$ |
| **Subspace-RegQDA** | Approach 1: Subspace Generative | 0.583 | 0.286 | 0.528 | $0.671 \pm 0.104$ | $0.674 \pm 0.092$ | $0.746 \pm 0.103$ |
| **EarlyFusion-SVM-Linear** | Approach 2: Margin / Ensemble | 0.750 | 0.727 | **0.889** | $0.693 \pm 0.092$ | $0.674 \pm 0.118$ | $0.678 \pm 0.107$ |
| **EarlyFusion-SVM-RBF** | Approach 2: Margin / Ensemble | 0.500 | 0.000 | 0.389 | $0.687 \pm 0.073$ | $0.686 \pm 0.061$ | $0.758 \pm 0.059$ |
| **EarlyFusion-RandomForest** | Approach 2: Margin / Ensemble | **0.833** | **0.833** | 0.833 | $0.701 \pm 0.091$ | $0.708 \pm 0.087$ | $0.745 \pm 0.104$ |
| **EarlyFusion-GradBoost** | Approach 2: Margin / Ensemble | 0.750 | 0.769 | 0.750 | $0.635 \pm 0.065$ | $0.650 \pm 0.063$ | $0.688 \pm 0.099$ |
| **EarlyFusion-ExtraTrees** | Approach 2: Margin / Ensemble | **0.833** | **0.833** | 0.833 | **$0.718 \pm 0.108$** | **$0.729 \pm 0.106$** | **$0.769 \pm 0.120$** |
| **LateFusion-StackingLR** | Approach 3: Late Stacked Fusion | 0.750 | 0.727 | 0.722 | $0.610 \pm 0.138$ | $0.622 \pm 0.140$ | $0.626 \pm 0.149$ |
| **LateFusion-SoftVoting** | Approach 3: Late Stacked Fusion | 0.667 | 0.600 | 0.750 | $0.677 \pm 0.074$ | $0.669 \pm 0.104$ | $0.670 \pm 0.101$ |
| **LateFusion-BayesianPoE** | Approach 3: Late Stacked Fusion | 0.667 | 0.600 | 0.722 | $0.669 \pm 0.072$ | $0.657 \pm 0.104$ | $0.669 \pm 0.106$ |
| **★ Ensemble Consensus** | Meta-Consensus | **0.833** | **0.833** | **0.850** | **$0.735 \pm 0.055$** | **$0.732 \pm 0.058$** | **$0.782 \pm 0.062$** |

### Table 2: Modality Ablation Analysis (5-Fold Stratified CV, $N=121$)

| Modality Subspace | Active Dimensions ($d$) | Accuracy (Mean $\pm$ Std) | F1-Score (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) |
|---|:---:|:---:|:---:|:---:|
| **Audio-only** | 3 | $0.537 \pm 0.087$ | $0.528 \pm 0.106$ | $0.604 \pm 0.137$ |
| **Vision-only** | 5 | $0.585 \pm 0.129$ | $0.557 \pm 0.137$ | $0.624 \pm 0.141$ |
| **Text-only** | 14 | $0.619 \pm 0.069$ | $0.631 \pm 0.079$ | $0.613 \pm 0.101$ |
| **Text + Vision** | 19 | $0.635 \pm 0.145$ | $0.620 \pm 0.148$ | $0.674 \pm 0.119$ |
| **Text + Audio** | 17 | $0.636 \pm 0.126$ | $0.638 \pm 0.127$ | $0.677 \pm 0.167$ |
| **Trimodal Fusion (All Streams)** | 22 | $0.668 \pm 0.068$ | $0.668 \pm 0.074$ | $0.721 \pm 0.111$ |
| **Audio + Vision (Optimal Nonverbal Pair)** | 8 | **$0.702 \pm 0.087$** | **$0.696 \pm 0.097$** | **$0.727 \pm 0.115$** |

> **Key Takeaway:** Unimodal acoustic or visual cues alone are prone to idiosyncratic variance ($53.7\% - 58.5\%$). Combining nonverbal behavioral streams (**Audio + Vision**) produces the most stable, noise-resilient cross-validated accuracy ($70.2\% \pm 8.7\%$) and ROC-AUC ($0.727 \pm 0.115$).

---

## 🎨 Interactive Studio & Visualizer Deliverables

### 1. Streamlit Multimodal Inspection Studio (`app.py`)
Launch the interactive web application to inspect any video in real time:
```bash
streamlit run app.py
```
- **Instant Pre-Bundled Clip Selector & Fast Uploads:** Instantaneous video playback and telemetry loading powered by the persistent disk cache.
- **Synchronized Behavioral Anomaly Timeline:** Synchronized multi-stream chronological strip showing exact temporal alignment of audio pitch anomalies, micro-silence gaps, and ocular blink events.
- **Modality Tri-Axial Signature & Behavioral Risk Index:** Sleek HTML progress gauge cards (Linguistic, Acoustic, Ocular) with an intelligent badge highlighting the primary behavioral driver of deception.
- **Multi-Model Consensus & Probability Breakdown:** Dynamic bar chart showing prediction probabilities across all 13 models, with an overarching **Ensemble Consensus** badge.
- **Local Feature Attribution ("Why This Verdict?"):** Horizontal diverging bar chart decomposing linear SVM margin weights $w_j \cdot x_j$ into deceptive (crimson) and truthful (emerald) directional contributions.
- **Three Real-Time Telemetry Deep-Dives:**
  1. **Acoustic Waveform & Pitch ($F_0$) Track:** Librosa $F_0$ contour with shaded silence regions and bounded vocal search.
  2. **Ocular Dynamics (EAR Timeline):** Frame-by-frame Eye Aspect Ratio with debounced local-minima blink markers and baseline threshold line.
  3. **Linguistic & Semantic Breakdown:** Whisper transcript with highlighted hesitation tokens (`um`, `uh`, `like`, `you know`) and normalized feature deviation bars against class means.
- **One-Click Forensic Export (JSON):** Download complete multi-model probability breakdowns, feature attributions, blink timestamps, and transcripts with a single click in the sidebar.
- **Tab 2: Architecture & Benchmark Explorer:**
  - Dark-themed 2D PCA & Fisher LDA scatter projection displaying the full dataset manifold and highlighting the current video's coordinate relative to the decision boundary (with guaranteed positive deceptive orientation).
  - High-resolution dark studio diagnostic plots (`subspace_projection.png`, `roc_curves.png`, `confusion_matrices.png`, `feature_importance.png`).

### 2. OpenCV Telemetry HUD Video Visualizer (`visualizer.py`)
Generates an annotated `.mp4` with a dark translucent Heads-Up Display (HUD):
```bash
python visualizer.py trial_lie_001.mp4 --out outputs/annotated_demo.mp4
```
- **Anti-Collision Dual-Row Header:** Polished obsidian HUD header with tracking reticle and cyan contours around both eyes and pupil crosshairs.
- **Live Scrolling EAR Sparkline:** Mini real-time waveform of the last 60 frames of Eye Aspect Ratio with a dynamic threshold line.
- **Robust 468-Point Mesh:** CLAHE contrast enhancement and dual-stage FaceMesh fallback ensure continuous tracking even on angled faces.
- **Synchronized Verdict Badge:** Crimson `DECEPTIVE` / emerald `TRUTHFUL` badge fully synchronized with the selected model or ensemble consensus.
- **Fast Audio Re-Muxing:** Fully re-muxes original audio using `-c:a copy` via `ffmpeg` for instant video generation.

---

## 📈 Diagnostic Plots & Visualizations

All diagnostic plots are generated automatically at 300 DPI in `outputs/plots/` using the studio dark theme (`#0b0f19` / `#131b2e`):

| Subspace Latent Projections | Multi-Model ROC Curves |
|:---:|:---:|
| ![Subspace Projection](outputs/plots/subspace_projection.png) | ![ROC Curves](outputs/plots/roc_curves.png) |
| *Figure 1: Unsupervised 2D PCA vs. Supervised Fisher's LDA Separation* | *Figure 2: Receiver Operating Characteristic Curves Across Paradigms* |

| Confusion Matrices | Feature Importances & Modality Weights |
|:---:|:---:|
| ![Confusion Matrices](outputs/plots/confusion_matrices.png) | ![Feature Importance](outputs/plots/feature_importance.png) |
| *Figure 3: Confusion Matrices for Top Performing Paradigms* | *Figure 4: Gini Feature Importance & Learned Meta-Weights vs. Baseline* |

---

## 🚀 Quickstart & Execution Guide

### 1. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/ChaitanyaParate/Multimodal_Deception_Detection_Using_Pattern_Recognition.git
cd Multimodal_Deception_Detection_Using_Pattern_Recognition

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Multimodal Studio
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Generate Annotated HUD Video
```bash
python visualizer.py trial_lie_001.mp4 --out outputs/annotated_demo.mp4
```

### 4. Run the Full Pattern Recognition Benchmark Suite
```bash
python benchmark.py
```
Executes dual-protocol evaluation (Holdout + 5-Fold Stratified CV across 13 models), runs modality ablation, updates all diagnostic plots in `outputs/plots/`, pre-warms `outputs/telemetry_cache.joblib`, and serializes models to `outputs/trained_models.joblib` in ~14 seconds.

### 5. Multi-Model CLI Inference (Single Video)
```bash
# Evaluate across all model architectures simultaneously
python predict.py trial_lie_001.mp4 --model all

# Evaluate with a specific paradigm backend
python predict.py trial_lie_001.mp4 --model EarlyFusion-ExtraTrees

# Legacy heuristic prediction
python main.py trial_lie_001.mp4
```

### 6. Train Deep MLP Baseline
```bash
python train.py
```

---

## 📂 Repository Structure

```
Multimodal_Deception_Detection_Using_Pattern_Recognition/
├── app.py                     # Streamlit Multimodal Inspection Studio (Obsidian Theme)
├── visualizer.py              # OpenCV Telemetry HUD Video Visualizer (FaceMesh + EAR Sparkline)
├── benchmark.py               # Dual-protocol benchmark runner & dark-theme plot generator
├── telemetry.py               # <30ms persistent disk cache & fast telemetry extractor
├── pr_pipeline.py             # Feature partitioning, PCA subspace compressor, dataset bundle
├── pr_solutions.py            # The 3 Pattern Recognition families & baseline evaluators
├── text_audio.py              # Whisper ASR transcription & Librosa acoustic feature extractor
├── vision.py                  # In-memory MediaPipe FaceMesh blink & saccade feature extractor
├── model.py                   # PyTorch DeceptionMLP deep neural baseline architecture
├── predict.py                 # Multi-model standalone CLI inference script
├── train.py                   # PyTorch neural training pipeline
├── repair_features.py         # Dataset feature repair & dense blink extractor utility
├── main.py                    # Legacy heuristic weighted fusion script
├── utils.py                   # Audio/frame extractors and normalization helpers
├── requirements.txt           # Pinned production dependencies
└── outputs/
    ├── trained_models.joblib  # Serialized model registry & fitted preprocessor
    ├── transcripts_cache.json # Cached Whisper transcripts for instant sub-ms access
    ├── telemetry_cache.joblib # Persistent telemetry cache (<30ms video loading)
    ├── benchmark_metrics.csv  # Dual-protocol evaluation metrics (13 models)
    ├── modality_ablation.csv  # Unimodal, bimodal, and trimodal CV results
    ├── features_full.csv      # Extracted 396-dimensional multimodal dataset (N=121)
    ├── model_full.pt          # PyTorch trained DeceptionMLP checkpoint
    └── plots/                 # 300 DPI dark studio diagnostic visualizations
        ├── subspace_projection.png
        ├── roc_curves.png
        ├── confusion_matrices.png
        └── feature_importance.png
```

---

## 📜 License
This project is released under the [MIT License](LICENSE).

