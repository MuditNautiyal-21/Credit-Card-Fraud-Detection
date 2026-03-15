# Credit Card Fraud Detection

Anomaly detection in credit card transactions using Isolation Forest and Autoencoder neural networks, trained without fraud labels to simulate real-world unsupervised detection.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## The Problem

Credit card fraud accounts for billions of dollars in losses annually. In practice, fraud labels are rare, delayed, or completely unavailable. Only 0.17% of transactions in a typical dataset are fraudulent, making supervised classification unreliable without heavy resampling. Financial institutions need detection systems that can flag anomalies without relying on pre-labeled fraud data.

## The Solution

This project builds and compares two unsupervised anomaly detection models on 284,807 real credit card transactions (492 fraudulent). Neither model sees fraud labels during training, simulating the real-world constraint where fraud patterns must be learned from transaction structure alone.

---

## Dataset

- 284,807 transactions with 492 confirmed fraud cases (0.17% positive rate)
- 30 features: 28 PCA-transformed components plus Time and Amount
- Sourced from the widely referenced Kaggle credit card fraud dataset

---

## Approach

1. Exploratory data analysis and feature scaling (StandardScaler on Amount and Time)
2. Trained two unsupervised models independently:
   - **Isolation Forest**: Tree-based algorithm that isolates anomalies by random partitioning. Fraudulent transactions require fewer splits to isolate, producing shorter path lengths.
   - **Autoencoder**: Neural network trained to reconstruct normal transactions. Fraud transactions produce higher reconstruction error because the model has never learned their patterns.
3. Evaluated both models on Precision, Recall, and F1-Score against held-out ground truth labels
4. Compared confusion matrices side by side to analyze detection tradeoffs

---

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 Score |
|-------|-------------------|----------------|----------|
| Isolation Forest | 25.77% | 25.41% | 25.59% |
| Autoencoder | 2.92% | 84.55% | 5.65% |

**Isolation Forest** is conservative: it flags fewer transactions overall, resulting in higher precision but lower recall. It misses more fraud but produces fewer false alarms.

**Autoencoder** is aggressive: it catches 84.55% of all fraud but at the cost of flagging many legitimate transactions. High recall, low precision.

The choice between models depends on the business context. If the cost of missing fraud is high (chargebacks, regulatory fines), the Autoencoder is preferable. If operational cost of investigating false positives matters more, Isolation Forest is the better fit.

---

## Key Takeaways

- Unsupervised models can detect fraud without any labeled training data, making them viable for cold-start scenarios
- Precision vs recall tradeoff is a business decision, not just a technical one
- An ensemble combining both models could balance detection coverage with false positive control
- Reconstruction error from the Autoencoder provides a continuous fraud probability score, useful for risk-tiered alerting

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Notebook | Jupyter |
| ML Models | Scikit-learn (Isolation Forest), TensorFlow/Keras (Autoencoder) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Scaling | StandardScaler |

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── fraud_detection.ipynb       # Full analysis, modeling, and evaluation
├── dataset_info.txt            # Dataset metadata and source
├── requirements.txt            # Python dependencies
├── images/                     # Visualizations and plots
├── LICENSE                     # MIT License
└── README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MuditNautiyal-21/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook fraud_detection.ipynb
```

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and placed in the root directory.

---

## Future Work

- Combine Isolation Forest and Autoencoder in a weighted ensemble for balanced precision and recall
- Apply SMOTE or other synthetic oversampling for supervised comparison baseline
- Deploy the model as a REST API for real-time transaction scoring
- Add SHAP or LIME explainability to surface which features drive fraud predictions

---

## License

MIT License. See [LICENSE](LICENSE) for details.
