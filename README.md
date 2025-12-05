# Supervised Learning Benchmark: Five Classifiers on Two Datasets

This project evaluates five classical supervised learning algorithms across two datasets (binary & multiclass) to compare model accuracy, training time, and behavior with respect to hyperparameter selection and learning curves.

The classifiers implemented:

- Decision Tree  
- Neural Network (MLPClassifier)  
- Support Vector Machine  
- AdaBoost  
- K-Nearest Neighbors  

The full analysis is documented in the final project report and reproduced here with runnable Python code.

---

## Final Project Report  
The complete methodology, figures, and analysis are described in the PDF report.

---

## Project Structure

```
project_root/
│
├── data/
│   ├── wisc_bc_data.csv            # Breast cancer dataset (binary classification)
│   └── wifi_localization.csv       # WiFi signal dataset (4-class classification)
│
├── ML_supervised_learning.py       # Unified script (breast + wifi + prefix system)
├── README.md                       # (this file)
└── results/                        # Generated plots saved automatically
```

---

## Datasets

### 1) Breast Cancer Wisconsin Dataset  
Binary classification dataset with 30 numerical features extracted from FNA images.

### 2) WiFi Localization Dataset  
Multiclass classification dataset (4 rooms) using WiFi signal strength from 7 access points.

---

## How to Run

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Select dataset inside the script:

```python
DATASET = 'breast'   # or 'wifi'
```

Run the script:

```bash
python ML_supervised_learning.py
```

Results will be saved automatically with prefixes:

```
[breast]DT-maxdepth.png
[wifi]SVM-learning-curve.png
...
```

---

## Key Findings

- Neural Network and SVM achieve the highest accuracy.  
- Decision Tree shows high variance (especially on smaller datasets).  
- AdaBoost is sensitive to learning rate and underperforms on multiclass WiFi data.  
- KNN improves significantly with larger datasets.

---
