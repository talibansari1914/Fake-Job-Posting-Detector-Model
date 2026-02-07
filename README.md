# 🕵️ Fake Job Posting Detector (Supervised ML)

A multi-model supervised machine learning project that detects **fake job postings**<br>
using Natural Language Processing (NLP) and classification algorithms.<br>

---

## Project Highlights

- Text classification using NLP<br>
- Multiple supervised ML models comparison<br>
- Clean preprocessing pipeline<br>
- Majority-voting based final prediction<br>
- Streamlit-based interactive UI (local)<br>

---

## Models Used

- Logistic Regression<br>
- Multinomial Naive Bayes<br>
- Random Forest Classifier<br>
- Linear Support Vector Classifier (SVC)<br>

Each model independently predicts whether a job posting is **Real or Fake**.<br>

---

## Tech Stack

- Python<br>
- Scikit-learn<br>
- NLTK<br>
- TF-IDF Vectorizer<br>
- Streamlit<br>
- Pickle<br>

---

## Model Evaluation

Models were evaluated offline using:<br>

- Accuracy<br>
- Precision<br>
- Recall<br>
- F1-score<br>
- Confusion Matrix<br>

Evaluation was performed on both **training and testing datasets**  <br>
inside the Jupyter Notebook.<br>

> Note: No evaluation logic is used in the deployed application.<br>

---

## Application Workflow

1. User enters job description<br>
2. Text is cleaned and vectorized<br>
3. Each ML model makes a prediction<br>
4. Predictions are displayed model-wise<br>
5. Final decision is made using **majority voting**<br>

---

## Project Structure

Fake-Job-Posting-Detector/<br>
│
├── multi_model_based_prediction/
|    ├── app.py<br>
|    ├── prediction_image.png<br>
|    ├── fake_job_detection_model<br>
├── single_model_based_prediction/<br>
|   ├── app.py<br>
|   ├── fake_job_detector_app<br>
|   ├── requirements.txt<br>
├── pickel_files<br>
|__fake_jobpostings.csv<br>
└── README.md<br>

---

## Features
- Text cleaning (lowercasing, stopword removal, stemming)<br>
- TF-IDF based feature extraction<br>
- Binary classification (Real vs Fake)<br>
- Prediction confidence using probability scores<br>
- Simple and interactive UI<br>

---

## Model Details
- Algorithm: Logistic Regression<br>
- Type: Supervised Classification<br>
- Evaluation Focus:<br>
    - Precision<br>
    - Recall<br>
    - F1-score<br>
    - Special attention given to minimizing false negatives (missing fake jobs)<br>

---

## How to Run Locally<br>
- streamlit run app.py`<br>

---

## Key Learning Outcomes

- End-to-end supervised ML workflow<br>
- Handling text data using NLP<br>
- Multi-model comparison and selection<br>
- ML inference using Streamlit<br>
- Clean separation of training and deployment logic<br>

---
## Future Improvements
- Improve UI with confidence bars<br>
- Deploy using Docker<br>
- Extend to multilingual job postings<br>

---

## 👤 Author
**Abbu Talib Ansari**<br>
GitHub:https://github.com/talibansari1914<br>
