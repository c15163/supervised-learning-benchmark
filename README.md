Supervised Learning Benchmark: Five Classifiers on Two Datasets

This project evaluates five classical supervised learning algorithms across two datasets (binary & multiclass) to compare model accuracy, training time, and behavior with respect to hyperparameter selection and learning curves.

The classifiers implemented:

Decision Tree

Neural Network (MLPClassifier)

Support Vector Machine

AdaBoost

K-Nearest Neighbors

The full analysis is documented in the final report and reproduced here with runnable Python code.

📄 Final Project Report:
The complete methodology, figures, and analysis are described in the PDF report.


📂 Project Structure
project_root/
│── data/
│     ├── wisc_bc_data.csv           # Breast cancer dataset (binary classification)
│     ├── wifi_localization.csv      # WiFi signal dataset (4-class classification)
│
│── ML_supervised_learning.py        # Unified script (breast + wifi + prefix system)
│── README.md                        # (this file)
│── results/                         # Generated plots saved automatically


The Python script automatically loads the correct dataset and names result files using prefixes:

DATASET 설정	prefix	예시 출력 파일
DATASET='breast'	[breast]	[breast]DT-maxdepth.png
DATASET='wifi'	[wifi]	[wifi]SVM-learning-curve.png
📊 Datasets
1) Breast Cancer Wisconsin (Diagnostic) Dataset

569 samples

30 numerical features extracted from FNA images

Binary classification (Benign vs Malignant)

Highly imbalanced, requiring normalization and careful hyperparameter tuning

The report (page 1–2) describes dataset motivation and preprocessing steps.


2) WiFi Localization Dataset

2,000 samples

Signal strengths from 7 WiFi access points

4-class indoor room classification

More complex, requires models that can separate multiclass boundaries

Report page 1 describes dataset rationale and structure.


⚙️ Implemented Classifiers

Each classifier undergoes the same rigorous evaluation:

✔ Hyperparameter search using validation curves
✔ 10-fold cross-validation
✔ Learning curves
✔ Training time measurement
✔ Accuracy comparison

This procedure is illustrated in Algorithm 1 from the report (page 1).


Decision Tree

Hyperparameters: max_depth, ccp_alpha

Observations:

Breast cancer: high variance due to limited samples (page 2–3)


WiFi: deeper trees permitted, lower variance

Neural Network (MLP)

Parameters examined: learning rate, α (regularization), hidden layer size

Best performance across both datasets

Low variance, small bias (page 4–5)


Support Vector Machine

Kernel selection (RBF, linear, sigmoid)

Best overall generalization when tuned properly

Breast cancer: RBF kernel best (page 6)


WiFi: finer gamma required for multi-class data (page 7)

AdaBoost

Very sensitive to learning rate & number of estimators

Performs well on binary classification

Struggles on multiclass WiFi dataset (page 8)


K-Nearest Neighbors

Hyperparameters: p (Minkowski metric), number of neighbors

Higher variance when the dataset is small

Much better performance on WiFi dataset due to larger sample size (page 9–10)


🧪 Summary of Performance

The report’s conclusion (page 11) shows:

Classifier	Breast Accuracy	WiFi Accuracy	Breast Time	WiFi Time
Decision Tree	94.15%	96.75%	0.008s	0.003s
Neural Network	98.83%	98.25%	2.63s	2.53s
SVM	98.83%	97.75%	0.005s	0.041s
AdaBoost	97.66%	94.75%	1.53s	0.10s
KNN	97.66%	97.88%	0.001s	0.003s

Neural Network and SVM achieve the best accuracy across both datasets.

Decision Tree is the weakest performer due to high variance in smaller datasets.

AdaBoost suffers on the multiclass WiFi dataset.

🚀 How to Run
Install Dependencies
pip install numpy pandas scikit-learn matplotlib

Run the unified script
python ML_supervised_learning.py


To switch between datasets:

DATASET = 'breast'   # or 'wifi'


All plots and comparison charts will be saved automatically in the project directory using the appropriate prefix.

📈 Generated Outputs

The script produces:

Validation curves

Learning curves

Loss curves (for NN)

Training time comparison

Accuracy comparison

These match the figures in the full report starting on page 2.


📝 References

References appear in the final report (page 11).

🎯 Key Takeaways

Neural Networks are the most accurate but computationally heavy

SVM offers excellent performance with fast training time

Decision Tree suffers most from dataset size (high variance)

AdaBoost requires careful tuning and struggles in multiclass settings

KNN improves substantially when the dataset size increases
