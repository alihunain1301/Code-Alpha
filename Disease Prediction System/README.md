# Disease Prediction System

A machine learning system to predict the presence of heart disease from clinical features. The project provides a Jupyter Notebook with data loading, preprocessing, model training and evaluation, model export, and a Gradio web interface for inference. This repository demonstrates model development and an interactive demo for non-technical users.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Results & Evaluation](#results--evaluation)
- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Run in Jupyter/Colab](#run-in-jupytercolab)
  - [Run the Gradio App](#run-the-gradio-app)
- [Project Structure](#project-structure)
- [How it works](#how-it-works)
  - [Preprocessing](#preprocessing)
  - [Models Trained](#models-trained)
  - [Model Selection and Saving](#model-selection-and-saving)
- [Usage examples](#usage-examples)
  - [Programmatic inference](#programmatic-inference)
  - [Sample input](#sample-input)
- [Reproducibility](#reproducibility)
- [Caveats & Notes](#caveats--notes)
- [Improvements & Next steps](#improvements--next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project trains several classification models to predict heart disease (binary target: disease vs no disease) from a tabular dataset (columns include age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, oldpeak, slope, CA, and thal). After training, the best model is saved along with the scaler, and a Gradio-based UI is provided for interactive predictions.

---

## Features

- End-to-end Jupyter Notebook for:
  - Data loading and exploration
  - Preprocessing (scaling)
  - Training multiple ML models (Logistic Regression, Random Forest, SVM, XGBoost)
  - Model evaluation and confusion matrix visualization
  - Saving model and scaler with `joblib`
  - Gradio UI for live inference
- Saved artifacts: `models/disease_model.pkl` and `models/scaler.pkl`
- Quick interactive demo via Gradio (useful for demos and user testing)

---

## Dataset

This repository expects a file named `heart.csv` in the repo root (or working directory) with the following columns:
- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

`target` is the binary label (0 = No disease, 1 = Disease). Ensure dataset is properly licensed for use.

---

## Results & Evaluation

The notebook trains four models and prints metrics on a stratified test split (test_size=0.2, random_state=42). Example results observed while developing:

- Logistic Regression: Accuracy ≈ 80.98%
- Random Forest: Accuracy = 100.00% (perfect on test set)
- SVM (RBF): Accuracy ≈ 92.68%
- XGBoost: Accuracy = 100.00% (perfect on test set)

NOTE: Perfect (100%) scores are unusual and may indicate overfitting, data leakage, or that the dataset/test-split contains duplicate or trivial examples. See the "Caveats & Notes" section.

---

## Quick Start

### Requirements

- Python 3.8+ recommended
- The main Python packages used:
  - numpy, pandas, scikit-learn, xgboost, joblib, matplotlib, seaborn, gradio
- Example `requirements.txt`:
