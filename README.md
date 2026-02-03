## Title:
# AnaVerse 2.0 — Temporal & Sensor-Based Anomaly Detection

This repository contains my final solution for the AnaVerse 2.0 Kaggle Hiring Challenge, where the goal was to detect anomalies in energy manufacturing plant sensor data using machine-learning models.

- Final Score: 0.8337 (F1)
- Final Rank: Top-20
- Primary Model: Random Forest with Threshold Optimization

## Problem Statement:

Sensor readings were collected at regular time intervals from an energy manufacturing plant.
The objective was to:
- Analyze the tabular sensor data
- Engineer temporal features from timestamps
- Handle extreme class imbalance
- Build robust predictive models
- Optimize performance using F1-score

## Approach Overview:

The workflow followed an end-to-end data-science pipeline:
- Problem understanding and objective definition
- Exploratory Data Analysis (EDA)
  - Univariate, bivariate, and multivariate analysis
- Target imbalance analysis
- Data preprocessing and feature engineering
- Baseline Logistic Regression model
- Random Forest modeling
- Threshold tuning for F1-score optimization
- Random seed experimentation
- Final model selection and retraining on full data
- Submission generation
- Conclusion

## Tools & Libraries:

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Final Model:

- Algorithm: Random Forest Classifier
- Imbalance Handling: class_weight="balanced"
- Optimized Threshold: 0.30
- Evaluation Metric: F1-score
- Validation-based tuning + full-data retraining

## Results Summary:

| Model                 | F1 Score | Notes                |
| --------------------- | -------- | -------------------- |
| Logistic Regression   | 0.10     | Baseline             |
| Random Forest         | 0.78     | Strong baseline      |
| Random Forest (tuned) | **0.83** | Final model          |
| LightGBM              | 0.44     | High false positives |
| XGBoost               | 0.44     | Similar behavior     |

## Dataset:

The dataset was provided through the AnaVerse 2.0 Kaggle Hiring Challenge.
Due to competition usage restrictions, the raw data is not included in this repository.

You can access the dataset directly from Kaggle:
=> https://www.kaggle.com/competitions/ana-verse-2-0-h/data

## Highlights:

- Top-20 finish in competitive hiring challenge
- Heavy class-imbalance handling
- Advanced threshold tuning
- Ensemble experimentation
- Production-ready ML workflow
