# Credit Scoring Model using LogisticRegressionCV

## 📋 Project Overview
This project focuses on building an end-to-end Machine Learning pipeline for credit risk assessment (credit underwriting). The primary objective is to predict whether a customer will repay their loan (`loan_paid_back`) based on their historical financial metrics and background attributes.

## 🛠️ Data Pipeline & Architecture
The project implements a strict data preprocessing architecture designed to prevent **Data Leakage** and ensure robust model generalization:
1. **Categorical Encoding:** Processed via Scikit-Learn's `OneHotEncoder` with native `Pandas DataFrame` output integration to eliminate multicollinearity and keep track of feature names.
2. **Validation Strategy:** The dataset is split into `train` and `test` sets *before* any feature scaling or transformation takes place.
3. **Feature Scaling:** `StandardScaler` is fitted (`.fit_transform()`) strictly on the training set and applied (`.transform()`) to the test set, mimicking real-world production deployment.
4. **Modeling:** Powered by `LogisticRegressionCV` with 5-fold cross-validation, directly optimizing the `ROC-AUC` scoring metric to find the ideal regularization strength ($C$).

## 📈 Results & Business Metrics
Instead of using the default 0.5 decision threshold, the mathematically optimal threshold was calculated using the **Youden's J statistic** ($J = TPR - FPR$) to maximize bank profitability and balance risk.

* **Overall Accuracy:** 86.3%
* **Precision (Class 1.0 - Reliable Clients):** 94% — *Minimizes default rates in the approved credit portfolio.*
* **Recall (Class 0.0 - Default Clients):** 77% — *The model successfully flags more than 3/4 of potential non-payers at the application stage.*

### Model Evaluation Visualizations

| ROC Curve (Separation Power) | Confusion Matrix (Business Metrics) |
|:---:|:---:|
| <img width="631" height="494" alt="image" src="https://github.com/user-attachments/assets/3946d5e9-1c09-4a80-ad59-a0c82583c997" /> | <img width="485" height="370" alt="image" src="https://github.com/user-attachments/assets/92859917-4cbc-4ff1-92c3-bf88758f5416" /> | |

### Confusion Matrix Breakdown:
* **True Positive (Correctly Approved):** 84,088
* **True Negative (Correctly Rejected Defaults):** 18,451
* **False Positive (Incorrect Approvals / Direct Loss):** 5,546
* **False Negative (Incorrect Rejections / Opportunity Cost):** 10,714


## ⚖️ Model Comparison & Selection

Two distinct machine learning algorithms were implemented, optimized using cross-validation, and evaluated on the same test set (118,799 samples).

| Model | Overall Accuracy | Recall (Class 0.0 - Default) | Precision (Class 0.0 - Default) | F1-Score (Class 0.0 / 1.0) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression CV** | 86.3% | **77.0%** | ~54.0% | 0.63 / 0.91 |
| **Random Forest Classifier** | **86.8%** | 76.0% | **65.0%** | **0.70 / 0.92** |

### 📈 Model Evaluation Insights
* **Random Forest Classifier (Top Performer)**: The ensemble model outperformed the linear baseline. While it maintains a highly competitive **76% Recall** for catching high-risk defaults (18,292 correctly flagged cases), it dramatically increases **Precision to 65%**. This significant drop in False Alarms directly translates to fewer accidental rejections of creditworthy applicants.
* **Logistic Regression CV (Interpretable Alternative)**: While slightly behind in overall predictive power, it remains a valuable asset for features dependency analysis and regulatory compliance due to its transparent log-odds coefficients.
* **Threshold Strategy**: For both models, the default `0.5` classification threshold was adjusted using **Youden's J statistic** ($J = TPR - FPR$). This mathematical approach optimized the trade-off between bank default protection and application conversion rates.


