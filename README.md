# Health Status Prediction Using Machine Learning in R

**Course / Subject:** Big Data & Machine Learning Project  
**Domain:** Healthcare Analytics & Preventive Medicine  

---

## 👥 Project Team Members
- **Hanna Sabik** – `ICE24AD028`
- **Rinu Fathima MS** – `ICE24AD055`
- **Hiba Fathima PS** – `ICE24AD031`
- **Fouzan Afeef** – `ICE24AD025`
- **Akhil Raj** – `ICE24AD007`

---

## 📌 Project Overview
This project develops an intelligent machine learning classification pipeline in **R** to predict an individual's **health risk status** (`Low Risk` vs. `High Risk`). By systematically analyzing multi-dimensional biometric markers and lifestyle behaviors, the system serves as an early-stage screening and preventive awareness tool.

> **Important Medical Disclaimer:**  
> The system is designed strictly as a supportive prediction and health awareness tool. It is not intended to replace professional medical diagnosis, clinical prescription, or doctor treatment.

---

## 🧬 Parameters Analyzed (13 Input Features)

| Category | Parameter | Description / Normal Baseline |
| :--- | :--- | :--- |
| **Demographics** | `age` | Patient age in years (18–80) |
| | `gender` | Male / Female biological stratification |
| **Biometric Vitals** | `systolic_bp` | Systolic Blood Pressure (<120 mmHg normal) |
| | `diastolic_bp` | Diastolic Blood Pressure (<80 mmHg normal) |
| | `blood_glucose` | Fasting Blood Sugar (70–99 mg/dL normal, >=126 diabetes) |
| | `bmi` | Body Mass Index (18.5–24.9 normal, >=30 obese) |
| | `heart_rate` | Resting pulse in BPM (60–100 normal) |
| | `cholesterol_total` | Total Serum Cholesterol (<200 mg/dL desirable) |
| **Lifestyle Habits** | `physical_activity_min` | Daily physical exercise in minutes |
| | `sleep_hours` | Average sleep duration per night (target 7–8 hrs) |
| | `smoking_status` | `Never`, `Former`, or `Current` tobacco smoker |
| | `alcohol_intake` | `None`, `Moderate`, or `Heavy` alcohol consumption |
| | `stress_level` | `Low`, `Medium`, or `High` self-reported stress |
| **Prediction Target** | `health_risk` | **`Low_Risk`** (Healthy) vs. **`High_Risk`** (Clinical Attention Advised) |

---

## 📊 Model Evaluation Results (Test Set Performance)

All models were evaluated on an independent 20% holdout test set (600 clinical records) using standard classification metrics:

| Classification Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(Best Model)* | **94.00%** | **94.79%** | **87.50%** | **91.00%** |
| **Support Vector Machine (SVM)** | **93.67%** | **93.37%** | **87.98%** | **90.59%** |
| **Random Forest (100 Trees)** | **91.00%** | **90.96%** | **82.21%** | **86.36%** |
| **Decision Tree (`rpart`)** | **81.50%** | **74.87%** | **70.19%** | **72.45%** |

---

## 🚀 How to Run the Project

### 1. Run the Complete ML Pipeline (Simple & Self-Contained)
Execute the single script to load data, train all 4 models, print confusion matrices, output metrics, generate the comparison chart, and run sample patient predictions:

```bash
cd /home/ashik/health_status_prediction
Rscript simple_health_prediction.R
```

### 2. Run Interactive Patient Prediction
Test custom patient parameters and receive instant risk assessment with clinical guidance:

```bash
cd /home/ashik/health_status_prediction
Rscript interactive_prediction.R
```

### 3. Open in RStudio
1. Open **RStudio**.
2. Set working directory: `Session -> Set Working Directory -> To Source File Location`.
3. Open `simple_health_prediction.R` and click **Source** or press `Ctrl + Shift + Enter`.

---

## 📽️ Presentation Files
The presentation has been generated in 16:9 widescreen format:

- **PowerPoint PPTX:** `Health_Status_Prediction_Presentation.pptx` (also available in `~/Downloads/`)
- **PDF Slides:** `Health_Status_Prediction_Presentation.pdf` (also available in `~/Downloads/`)

### Slide Structure (12 Slides):
1. **Title Slide:** Dark Navy theme, Big Data & ML tags, Team Member cards with registration numbers.
2. **Project Abstract:** Executive summary, key points, and summary metrics.
3. **Problem Statement & Objectives:** Asymptomatic progression, multi-factor complexity, project aims.
4. **Parameters Considered:** Demographic, Biometric vitals, and Lifestyle habit categorization.
5. **System Architecture & Workflow:** 5 distinct stages (Collection, Preprocessing, Training, Evaluation, Prediction).
6. **Why R Programming?:** Statistical native foundation, ML CRAN ecosystem, reproducible scripting.
7. **Machine Learning Classification Models:** Technical comparison of Logistic Regression, Decision Tree, Random Forest, and SVM.
8. **Model Evaluation & Performance Results:** Comprehensive metrics table highlighting Logistic Regression's 94% accuracy.
9. **Visualizations & Feature Insights:** Embedded performance bar chart and primary clinical drivers analysis.
10. **Live Demonstration:** High-risk vs. Low-risk patient profile walkthrough with probability scores.
11. **Benefits & Important Medical Disclaimer:** Preventive utility and ethical clinical boundary.
12. **Conclusion & Future Scope:** Achievements, smartwatch/IoT roadmap, multi-class disease extensions.
