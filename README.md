# Academic Dropout Risk Analysis

Machine learning analysis and predictive modeling to identify key factors and early risk of student dropout in higher education.

> **Status:** work in progress. Exploratory analysis is complete; modeling is in progress.

---

## 🎯 Problem Statement & Objectives

> **Core Question:** What key factors allow for early identification of students at risk of dropping out, and how accurately can this dropout be predicted?

### Project Goals:
1. **Descriptive Analysis (EDA):** Identify the primary socio-economic, academic, and behavioral factors associated with student attrition.
2. **Predictive Modeling:** Develop and evaluate machine learning models to estimate individual student dropout risk.

---

## 📊 Dataset

- **Source:** [Student Academic Performance & Dropout Risk 2026](https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk) (Kaggle).
- **Size:** 100,000 students, 32 features, a student identifier and the target `dropout` (1 = dropped out, 32% of students).
- **Feature groups:** demographics, socioeconomic, institutional, academic performance, engagement and behaviour, and wellbeing and support.
- **Important:** according to the dataset description, all records are **fully synthetic** and `dropout` was generated with a logistic model calibrated to published research. The results therefore illustrate the methodology and should not be read as findings about real students.

---

## 🔍 Key Findings (EDA)

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

- **Academic performance has the strongest association with dropout.** Median semester-1 GPA is 2.87 for students who stayed vs. 1.92 for those who dropped out (correlation with dropout: -0.57).
- **Unit pass rate** (units passed / units enrolled) is also strongly associated (correlation -0.51): dropout falls from about 88% (pass rate ≤50%) to 7% (>90%). It overlaps strongly with GPA.
- **Attendance** remains associated with dropout among students with similar grades: from about 59% (attendance ≤50%) to 16% (>90%) overall.
- **Financial background:** dropout is 42.9% for low-income, 30.0% for middle-income and 20.0% for high-income students, and the gap persists within GPA groups. Students without up-to-date tuition (44.2% vs. 28.7%) and with a part-time job (39.8% vs. 24.7%) also show higher dropout.
- **Little or no difference** by gender, nationality, course type or international status.

All findings are descriptive associations, not causal effects.

---

## 🤖 Modeling (in progress)

Notebook: [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb)

**Scenario A (early warning).** The model uses 26 of the 32 features as a design decision: `gpa_semester_2` and the variables describing the current semester (`semester`, `units_enrolled`, `units_passed`, `units_failed`, `missed_deadlines`) are excluded. The dataset does not document when the remaining variables are recorded, so their availability after the first semester is an assumption.

**Method:** stratified 80/20 train/test split; models compared with 5-fold stratified cross-validation on the training set; the test set is used once at the end.

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic regression | 26 (Scenario A) | 0.896 | 0.817 |
| Gradient boosting | 26 (Scenario A) | 0.893 | 0.812 |
| Logistic regression | all 32 | 0.904 | 0.831 |

The logistic regression with the 26 Scenario A features was selected: gradient boosting did not improve on it, and the excluded variables add only a small gain.

**Test set (logistic regression, 26 features):** ROC-AUC 0.896 and PR-AUC 0.818. A decision threshold of 0.30 was chosen on the training set with the criterion of detecting at least 80% of dropouts. On the test set it detects 83.0% of dropouts (recall) with a precision of 65.7%, flagging 40.4% of students as at risk.

**Model interpretation:** the coefficients of the logistic regression are consistent with the EDA: semester-1 GPA is the dominant factor (one standard deviation more is associated with odds of dropout about 0.19 times as high), followed by attendance, tuition status, weekly work hours, LMS activity, assignment submission and stress. Family income shows a clear gradient (low-income students have about 2.3 times the odds of high-income students, all else equal), while most demographic and institutional variables add little. These are conditional associations in synthetic data, not causal effects.
---

## 📁 Repository Structure

```
├── data/
│   ├── raw/              # original dataset (see Dataset section)
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb      # exploratory data analysis
│   └── 02_modeling.ipynb # predictive modeling
├── reports/figures/      # charts used in the analysis
├── src/
└── requirements.txt
```

---

## ⚙️ How to Reproduce

```bash
conda create -n dropout-risk python=3.11
conda activate dropout-risk
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the CSV in `data/raw/` as `student_dropout_risk_2026.csv`, then run the notebooks in order.

---

## ⚠️ Limitations

- The dataset is synthetic, so the results illustrate the methodology and not real-world performance.
- `dropout` has no date and most variables do not specify the period they refer to, so the "early warning" setting is a design choice that cannot be verified with this data.
- Associations are descriptive; they do not establish causation.