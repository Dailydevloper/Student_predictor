# Student Performance Prediction System

## Overview

This project predicts students' final grades (G3) using demographic, social, and academic features.

## Dataset

UCI Student Performance Dataset (Math Course)

## Models Used

- Linear Regression
- Random Forest Regressor
- Neural Network (TensorFlow)

## Methodology

- Data cleaning and EDA
- One-hot encoding
- Train-test split
- Feature importance analysis
- Model comparison

## Results

| Model             | RMSE | R2    |
| ----------------- | ---- | ----- |
| Linear Regression | ~4.2 | ~0.14 |
| Random Forest     | ~3.8 | ~0.27 |
| Neural Network    | ~5.1 | -0.27 |

Random Forest performed best on this dataset.

## Key Insights

- Attendance and past failures are strongest predictors
- Deep learning underperforms on small tabular data
- Removing grade leakage reduced performance but improved realism

## Tech Stack

Python, Pandas, Scikit-learn, TensorFlow, Matplotlib

## How to Run

cd src
python train.py
