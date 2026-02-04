# EEG Emotion Recognition with CNN–Transformer–LSTM Models

This repository demonstrates my approach to EEG-based emotion recognition using
deep learning architectures for time-series decoding. The focus of this codebase
is on model design, training, and subject-wise evaluation rather than raw signal
acquisition.

## Overview

The pipeline supports EEG feature inputs from public datasets such as SEED-IV and
DEAP and evaluates subject-wise classification performance using custom neural
network architectures.

## Pipeline

EEG Features (.mat)  
→ Dataset Loader & Normalization  
→ CNN-based Temporal Feature Extraction  
→ Transformer Encoder (Temporal Attention)  
→ Optional BiLSTM Layers  
→ Classification Head  

## Model Architecture

The primary model implemented in this repository is **ConTL**, which combines:

- **1D Convolutional layers** for local temporal pattern extraction  
- **Transformer encoder** to model long-range temporal dependencies  
- **Optional bidirectional LSTM layers** for sequential modeling  
- **Fully connected layers** for final classification  

The architecture is configurable via command-line arguments, allowing flexible
experimentation with different modeling choices.

## Datasets

The codebase supports:

- **SEED-IV**  
  - 4-class emotion classification  
  - Subject-wise evaluation (15 subjects)

- **DEAP**  
  - Valence-based emotion classification  

This repository assumes that EEG features (e.g., differential entropy across
frequency bands) have already been extracted. The emphasis here is on downstream
modeling and evaluation.

## Evaluation Strategy

- Subject-wise training and testing
- Train / validation / test splits per subject
- Early stopping with patience-based scheduling
- Performance metrics:
  - Classification accuracy
  - Weighted F1-score

Subject-wise accuracies are saved to CSV files for robustness analysis.


## Running the Code

Example command for SEED-IV:

```bash
python main.py \
  --model_type ConTL \
  --data-choice 4 \
  --data-path path_to_SEED_IV_features \
  --save_file_name seed4_results.csv

```
## Code Structure

.
├── main.py # Entry point for training and evaluation
├── solver.py # Training, validation, and testing logic
├── models.py # CNN–Transformer–LSTM model definitions
├── config.py # Experiment configuration
├── utils/
│ └── tools.py # Dataset loading and preprocessing utilities
├── result/
│ └── *.csv # Subject-wise evaluation results


