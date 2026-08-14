
# 🧬 Synthetic Medical Data Generator

A data-driven project for generating synthetic medical datasets while preserving important statistical and structural characteristics of the original data.

The project explores multiple synthetic data generation approaches and compares their performance using statistical similarity, machine learning utility, and evaluation metrics.

---

## 🎯 Project Objective

The main objective of this project is to investigate how synthetic medical data can be generated and evaluated for research and machine learning workflows.

The project focuses on:

- Generating realistic synthetic medical records
- Comparing multiple synthetic data generation techniques
- Evaluating statistical similarity between real and synthetic data
- Evaluating machine learning utility
- Analyzing the quality of generated synthetic datasets
- Exploring privacy-aware synthetic data generation

---

## 🧠 Synthetic Data Generation Methods

This project experiments with several synthetic data generation techniques:

- **CTAB-GAN**
- **CTGAN**
- **CopulaGAN**
- **Gaussian Copula**
- **TVAE**
- **medGAN**

The generated datasets are stored separately in the `synthetic_data/` directory.

---

## 🔬 Methodology

The overall workflow of the project is:

```text
Original Medical Dataset
          ↓
Data Preprocessing
          ↓
Data Cleaning & Transformation
          ↓
Synthetic Data Generation
          ↓
Multiple Generative Models
          ↓
Synthetic Dataset Evaluation
          ↓
Model Comparison
          ↓
Best Model Selection
