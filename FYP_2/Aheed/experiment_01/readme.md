## EEG and Facial Feature Fusion for Emotion Recognition

### Overview
This research explores the fusion of EEG signals with facial features for emotion recognition, utilizing the DEAP dataset. The study aims to integrate deep learning-based facial feature extraction with EEG data processing to enhance emotion classification accuracy.

### Dataset
- **EEG Data**: Extracted from the DEAP dataset, preprocessed into a shape of (32, 40, 40, 160), representing 32 participants.
- **Facial Data**: Available for only the first 22 participants, extracted from corresponding video recordings.

### Pipeline

#### 1. **Facial Feature Extraction**
- **Face Detection & Alignment**: Implemented using [Pytorch RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) to detect and align faces.
- Extracted deep facial features using **Vision Transformer (ViT)**.
- Output: A feature vector of shape **(1, 768)** for each detected face.

#### 2. **EEG Data Processing**
- EEG signals preprocessed into a structured format for deep learning.
- Potential feature extraction using CNN/RNN-based models (to be refined further).

#### 3. **Fusion Strategy**
- Combining extracted **facial feature embeddings** with **EEG embeddings**.
- Investigating **CLISA and Multi-CLISA** frameworks for fusion.

### Next Steps
- Batch process **video frames** to extract consistent facial features.
- Design **fusion model** to combine EEG and facial embeddings.
- Train and evaluate models on **emotion classification tasks**.

### Requirements
Ensure the following dependencies are installed:
```bash
pip install torch torchvision transformers opencv-python numpy scipy matplotlib
```
and clone 
```bash
https://github.com/biubug6/Pytorch_Retinaface.git
```
for running the files

### Execution
Run each module in sequence:
```bash
python detect.py   # Run RetinaFace for face detection
python vit.py      # Extract facial features using ViT
```
(Fusion and classification steps will be implemented in future iterations.)

### Research Goal
To develop a **robust multimodal emotion recognition system** by fusing **facial and EEG-based features**, leveraging deep learning techniques.

### Credits
- **Face Detection & Alignment**: [Pytorch RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- **Vision Transformer Model**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- **EEG Data Source**: [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

