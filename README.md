# 🧠 EEG-based Emotion Recognition

<div align="center">

![Brain Waves](https://img.shields.io/badge/🧠-EEG%20Signals-blueviolet?style=for-the-badge)
![Emotions](https://img.shields.io/badge/😀😢-Emotion%20Recognition-ff69b4?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

> *Decoding emotions from brain signals using cutting-edge contrastive learning and multimodal fusion*

## 🌟 Overview

This repository contains a comprehensive implementation of novel approaches for emotion recognition using electroencephalogram (EEG) signals and facial expressions. Our research focuses on solving one of the most challenging problems in affective computing: creating subject-invariant emotion recognition models that generalize across individuals.

### 🔍 Key Innovations

- **Contrastive Learning of Subject-Invariant EEG Representations (CLISA)** 
- **Multi-Source Contrastive Learning (MSCL)** for enhanced cross-subject generalization
- **Multimodal Fusion** combining EEG and facial expressions for robust emotion detection
- **Quantum Machine Learning Approaches** exploring quantum computing for emotion classification

## 🚀 Journey from FYP-1 to FYP-2

### 🏁 FYP-1: Foundation

In our initial phase, we focused on implementing the CLISA framework that leverages contrastive learning to create subject-invariant EEG representations. This approach demonstrated promising results but had limitations in handling noisy labels and domain adaptation.

#### Key Achievements:
- Successfully implemented the CLISA framework for the SEED dataset
- Achieved significant improvements over traditional approaches
- Established a preprocessing pipeline for EEG signals

### 🔬 FYP-2: Advanced Approaches

Building on our foundation, we developed three advanced approaches:

1. **DEAP EEG+Face Fusion (Multimodal)** by Aheed: Combining EEG signals with facial expressions for enhanced accuracy
2. **DEAP EEG-only Multi-class Classification** by Hamza: Improved subject-invariant features using Multi-Source Contrastive Learning
3. **SEED EEG-only Multi-class Classification** by Haseeb: Advanced noise-robust learning with Generalized Cross-Entropy loss

#### Key Improvements:
- Dynamic domain adaptation to enhance transferability across subjects
- Prototype embeddings to reduce the impact of noisy labels
- Dual-stage contrastive learning for better feature representation

## 🧩 Project Structure

```
├── FYP_1/                  # First phase implementations
│   ├── Version 1/          # Initial CLISA implementation
│   ├── Version 2_Hamza/    # Improved CLISA variants
│   ├── Version 2_Haseeb/   # Alternative implementations
│   ├── Version_Aheed/      # Additional experiments
│   └── Quantum_Approach/   # Quantum computing experiments
│
├── FYP_2/                  # Second phase implementations
│   ├── Aheed/              # Multimodal fusion implementation
│   │   ├── Face_Model/     # Facial expression models
│   │   └── Last/           # Final model implementations
│   ├── Hamza/              # DEAP dataset experiments
│   └── Haseeb/             # SEED dataset experiments
│
├── analysis_results/       # Visualizations and analysis reports
├── validation_results/     # Data validation outputs
└── Documentation/          # Project reports and presentations
```

## 🧪 Approaches & Results

### 1️⃣ DEAP EEG+Face Multimodal Fusion (Aheed)

Combines EEG signals with facial expressions to create a robust emotion recognition pipeline.

<div align="center">
  <img src="FYP_2/Aheed/Documentation/deaparc1.png" alt="EEG+Face Architecture" width="800"/>
  <p><i>Architecture of the DEAP EEG+Face fusion model, showing the separate encoders, fusion mechanism, and projection heads</i></p>
</div>

<div align="center">
  <img src="FYP_2/Aheed/Documentation/tsne_fused_by_emotion.png" alt="t-SNE of EEG+Face Fusion" width="400"/>
  <p><i>t-SNE visualization of the multimodal fused representations, showing the separation of emotion classes</i></p>
</div>

**Results:**
- Valence Classification Accuracy: 
  - Subject-Dependent: 68.37%
  - Subject-Independent: 54.55%
- F1-Score: 
  - Subject-Dependent: 63.02%
  - Subject-Independent: 53.87%
- Multimodal fusion provided complementary information, improving over EEG-only approaches

### 2️⃣ Multi-Source Contrastive Learning (Hamza)

Enhances cross-subject generalization through dynamic domain adaptation and noise-robust learning.

<div align="center">
  <img src="FYP_2/Aheed/Documentation/deaparc2.png" alt="DEAP EEG-only Architecture" width="800"/>
  <p><i>Architecture of the DEAP EEG-only model with subject-specific mapper and cross-subject alignment module</i></p>
</div>

<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="flex: 1; text-align: center;">
      <img src="FYP_2/Aheed/Documentation/beforedeap.png" alt="Before Subject Mapping" width="350"/>
      <p><i>Before Subject-Specific Mapper</i></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="FYP_2/Aheed/Documentation/afterdeap.png" alt="After Subject Mapping" width="350"/>
      <p><i>After Subject-Specific Mapper</i></p>
    </div>
  </div>
  <p><i>t-SNE visualizations showing the improvement in class separation after applying subject-specific mapping</i></p>
</div>

**Results:**
- DEAP Dataset: 85.4% (subject-dependent), 68.7% (subject-independent) accuracy
- Macro F1-score: 83.8% (subject-dependent), 63.2% (subject-independent)
- UAR: 82.7% (subject-dependent), 61.8% (subject-independent)
- 10.5% improvement over baseline methods

### 3️⃣ SEED EEG-only Multi-class Classification (Haseeb)

Advanced noise-robust learning with Generalized Cross-Entropy loss for improved emotion recognition.

<div align="center">
  <img src="FYP_2/Aheed/Documentation/seedarc1.png" alt="SEED EEG-only Architecture" width="800"/>
  <p><i>Architecture of the SEED EEG-only model with advanced loss functions</i></p>
</div>

<div align="center">
  <img src="FYP_2/Aheed/Documentation/afterseed.png" alt="SEED Embeddings" width="400"/>
  <p><i>t-SNE visualization of the SEED embeddings after training with the combined Prototype Contrastive and MMD loss</i></p>
</div>

**Results:**
- SEED Dataset: 86.48% average accuracy (3-class)
- Weighted F1-score: 86.56%
- Combined Prototype Contrastive and MMD loss significantly outperformed other loss functions
- Comparable to CLISA, with better handling of noisy labels

### 4️⃣ Quantum Machine Learning Approach (FYP-1)

Explores quantum computing for emotion classification using quantum neural networks.

**Features:**
- Quantum circuit-based feature extraction
- Hybrid quantum-classical model architecture
- Parameter-efficient quantum models

## 🔬 Technical Innovations

Our research introduced several key technical innovations:

<div align="center">
  <table>
    <tr>
      <th>Approach</th>
      <th>Key Technical Innovations</th>
    </tr>
    <tr>
      <td>DEAP EEG+Face (Aheed)</td>
      <td>
        <ul>
          <li>Cross-modal contrastive learning for aligning EEG and facial representations</li>
          <li>Multimodal fusion with attention mechanism</li>
          <li>Dynamic confidence-based weighting for each modality</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>DEAP EEG-only (Hamza)</td>
      <td>
        <ul>
          <li>Subject-specific mapper with Squeeze-and-Excitation blocks</li>
          <li>Cross-subject alignment module</li>
          <li>Dynamic Weighted Focal Loss for addressing class imbalance</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>SEED EEG-only (Haseeb)</td>
      <td>
        <ul>
          <li>Prototype Contrastive Loss for stable representation learning</li>
          <li>Maximum Mean Discrepancy (MMD) Loss for domain adaptation</li>
          <li>Momentum-based prototype updating mechanism</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

## 🔧 Installation & Usage

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/EEg-based-Emotion-Recognition.git
cd EEg-based-Emotion-Recognition

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# DEAP EEG+Face approach
python FYP_2/Aheed/Last/models/src/run_advanced_training.py

# SEED EEG-only approach
python FYP_2/Haseeb/SEED-FINAL_APPROACH/train.py

# Analysis and visualization
python analyze_model.py
python tsnevisual.py
```

## 📊 Performance

| Approach | Dataset | Subject-Dependent Accuracy | Subject-Independent Accuracy | F1-Score | UAR |
|----------|---------|----------|----------|----------|-----|
| CLISA (FYP-1) | SEED | N/A | 85.4% | N/A | N/A |
| MSCL (FYP-2) | SEED | N/A | 86.48% | 86.56% | N/A |
| EEG+Face (FYP-2) | DEAP | 68.37% | 54.55% | 53.87% | N/A |
| DEAP EEG-only (FYP-2) | DEAP | 85.4% | 68.7% | 63.2% | 61.8% |

## 👥 Team

- **Muhammad Hamza** (21k-3815)
- **Aheed Tahir Ali** (21k-4517)
- **Abdul Haseeb Dharwarwala** (21k-3217)

## 🔗 References

- X. Shen et al., "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition," in IEEE Transactions on Affective Computing, 2022.
- X. Deng et al., "Multi-Source Contrastive Learning for Cross-Subject EEG Emotion Recognition," in Biomedical Signal Processing and Control, 2024.
- Zheng et al., "Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks," in IEEE Transactions on Autonomous Mental Development, 2015.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <i>Made with 🧠 at FAST-National University of Computer & Emerging Sciences, Karachi</i>
</div>

<div align="center">
  <img src="FYP_2/Aheed/Documentation/tsne_comparison_report_style.png" alt="tSNE Visualization" width="600"/>
</div> 