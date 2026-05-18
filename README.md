# Human Activity Recognition — UCI-HAR Dataset

A multi-phase machine learning project for classifying human activities from smartphone sensor data. The notebook covers the full pipeline from raw data preprocessing through classical ML baselines and deep sequential models, ending with a fine-tuned time-series Transformer.

---

## Dataset

**[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)**

- **30 subjects**, activities recorded via waist-mounted Samsung Galaxy S II
- **561 hand-crafted features** from accelerometer and gyroscope signals (time & frequency domains)
- **Raw sensor signals** (9 channels × 128 timesteps) used for deep learning phases
- **6 activity classes:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

---

## Project Structure

```
notebook.ipynb
├── Phase 1 — Data Preprocessing & EDA
│   ├── Data loading & cleaning (duplicates, nulls, feature renaming)
│   ├── Exploratory visualizations (accelerometer & gyroscope signals)
│   └── Time-domain and frequency-domain analysis
│
├── Phase 2 — Classical ML & 1D-CNN Baselines
│   ├── K-Nearest Neighbors (KNN)
│   ├── Gaussian Naïve Bayes
│   ├── Random Forest & Gradient Boosting
│   └── 1D Convolutional Neural Network (CNN)
│
└── Phase 3 — Sequential Deep Learning Models
    ├── Vanilla RNN
    ├── GRU (Gated Recurrent Unit)
    ├── LSTM (Long Short-Term Memory)
    └── PatchTST Transformer (fine-tuned from ibm-granite/granite-timeseries-patchtst)
```

---

## Models & Evaluation

All models are evaluated on **accuracy**, **weighted F1-score**, and **per-class F1**. Confusion matrices and training curves are plotted for each deep learning model.

| Model | Input Type |
|---|---|
| KNN | Hand-crafted features |
| Gaussian Naïve Bayes | Hand-crafted features |
| Random Forest | Hand-crafted features |
| Gradient Boosting | Hand-crafted features |
| 1D-CNN | Raw sequences (128 × 9) |
| Vanilla RNN | Raw sequences (128 × 9) |
| GRU | Raw sequences (128 × 9) |
| LSTM | Raw sequences (128 × 9) |
| PatchTST Transformer | Raw sequences (128 × 9) |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
pip install transformers evaluate accelerate datasets
```

> **Note:** The notebook is designed to run on **Google Colab** (GPU recommended for the Transformer fine-tuning phase). The dataset is downloaded automatically via `wget`.

---

## Getting Started

1. Open the notebook in Google Colab.
2. Run the dataset download cells — the UCI-HAR zip is fetched and extracted automatically.
3. Execute phases sequentially. Each phase builds on preprocessed data from the previous one.

---

## Key Techniques

- Duplicate feature name resolution for the UCI-HAR 561-feature set
- Time & frequency domain signal visualization (accelerometer + gyroscope)
- Raw signal reshaping to `(samples, 128 timesteps, 9 channels)` for sequential models
- Early stopping with `restore_best_weights` across all deep learning models
- PatchTST fine-tuning via HuggingFace `Trainer` API with subject-based validation split
- Final cross-model comparison: accuracy, weighted F1, and training time
