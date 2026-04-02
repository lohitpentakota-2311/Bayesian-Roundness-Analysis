# 📊 Bayesian Models for Roundness Prediction

This repository contains a MATLAB implementation of Bayesian and regularized regression models for predicting **roundness error (μm)** in centreless grinding processes.

---

## 👨‍🔬 Authors
- **Lohit Kumar Pentakota**
- **Marco Leonesio** (Corresponding author)
- **Moschos Papananias**
- **Giacomo Bianchi**

📅 **Date:** March 31, 2026

---

## 📁 Repository Structure

├── data/  
│   └── InputData.mat  
├── utils/  
├── main_script.m  
└── README.md  

---

## ⚙️ Parameters & Requirements
- **MATLAB Toolboxes:** Statistics and Machine Learning, Econometrics.
- `seed = 1`: Ensures reproducibility of MCMC draws and CV splits.
- `FlagforParamterIdentification = 1`: Enables automated hyperparameter tuning (Lambda, V1, V2).
- `epsilon = 2.8`: The **Hybrid Error (MEHE)** threshold, capturing deviations critical to engineering tolerances.

---

## 🚀 How to Run

1. Place the dataset:
   /data/InputData.mat  

2. Make sure `/utils` folder is present  

3. Run in MATLAB:
   main_script  

---

## 📦 Dataset Requirements

The dataset must contain:

- `x` → feature matrix  
- `y` → target (roundness)  
- `FeatureNames` → predictor names  

---

## 🔬 Methodology

### 1. Data Preprocessing
- 14% hold-out test split  
- Data leakage prevention (duplicate process parameters removed from test set)  
- Feature normalization  

---

### 2. Bayesian LASSO
- Model: `bayeslm(...,'Lasso')`  
- Lambda optimized via cross-validation  
- MCMC Convergence estimation  
- Outputs used for:
  - Predictions  
  - RMSE  
  - Hybrid Error  
  - Prediction uncertainty

---

### 3. Bayesian Mix- Conjugate Model (SSVS)
- Spike-and-slab variable selection for feature importance analysis
- Posterior Inclusion Probabilities (PIP) calculated  
- Hyperparameters optimized  
- Outputs used for:
  - Predictions
  - RMSE 
  - Hybrid Error  
  - Feature importance  
  - Prediction uncertainty 
  - Subset analysis for minimal models 

### 4. Feature Selection & Minimal Models

#### Correlation Clustering
- Features grouped using correlation threshold  

#### Representative Selection
- Highest PIP feature per cluster selected  

#### PIP Threshold Optimization
- Grid search to select best subset  

---

### 5. Ridge Regression
- Applied to selected features  
- Lambda optimized via cross-validation  
- Provides:
  - Regression Coefficients  
  - Confidence intervals  
  - Prediction intervals  

---

## 📊 Evaluation Metrics

### RMSE
Standard error metric  

### Hybrid Error (MEHE)
- Uses threshold `epsilon = 2.8`  
- Captures engineering-relevant deviations  

---

## 📈 Outputs

- Predicted vs measured plots  
- Residual analysis  
- Confidence intervals  
- Prediction intervals
- Learning curves  
- RMSE vs correlation threshold  
- Hyperparameter heatmaps (PIP vs CorrCutOff)  

---

## 🔁 Reproducibility

- Fixed random seed ensures repeatable results

---

## ⚠️ Notes

- Ensure `/utils` is in MATLAB path  
- Hyperparameter tuning can be disabled by:
  FlagforParamterIdentification = 0  

---

## 📜 License

CC BY-NC-ND

---

## 📚 Citation

Pentakota, L. K., Leonesio, M.*, Papananias, M., & Bianchi, G., Matlab implementation: Monitoring Centreless Grinding Processes for Workpiece Roundness Prediction Using Internal Machine Sensors and Bayesian Sparse Regression Models, Version 1, GitHub, 2026. https://github.com/lohitpentakota-2311/Bayesian-Roundness-Analysis
