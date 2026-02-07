import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# stop words used such as : i me you will would etc
nltk.download('stopwords')

# Loading model & Vectorizer
lr_model = pickle.load(open("lr_model.pkl", "rb"))
mn_nb_model = pickle.load(open("mn_nb_model.pkl", "rb"))
rfc_model = pickle.load(open("rfc_model.pkl", "rb"))
svc_model = pickle.load(open("svc_model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# port stemmer initialization
stem_port=PorterStemmer()

def clean_text(text):
    text=re.sub('[^a-zA-Z]', ' ' ,text) # choosing only english characters
    text=text.lower() # cchanging it into the lower case letters
    text=text.split() # splitting text into the list of words
    text=[stem_port.stem(word) for word in text if not word in stopwords.words('english')]
    text= ' ' .join(text)
    return text



def predict_model(model, vectorized, has_proba=True):
    pred = model.predict(vectorized)[0]

    if has_proba:
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100
    else:
        confidence = None

    return pred, confidence


# UI For APP
st.set_page_config(page_title='Fake Job Detector',layout='centered')

# title
st.title('Fake job Posting Detector')

# write
st.write('Paste job description below to check authenticity')

# user input about the job
user_input=st.text_area('Job Description',height=200)

if st.button('Check Job'):
    if user_input.strip() == '':
        st.warning('Please enter job description')
    else:
        # 1️ preprocessing
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        # 2️ predictions
        lr_pred, lr_conf = predict_model(lr_model, vectorized)
        mn_nb_pred, mn_nb_conf = predict_model(mn_nb_model, vectorized)
        rfc_pred, rfc_conf = predict_model(rfc_model, vectorized)
        svc_pred, _ = predict_model(svc_model, vectorized, has_proba=False)

        # 3️. model-wise UI
        st.subheader(" Model-wise Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"Logistic Regression: {'Fake' if lr_pred else 'Real'}")
            st.write(f"Confidence: {lr_conf:.2f}%")

            st.info(f"Multinomial NB: {'Fake' if mn_nb_pred else 'Real'}")
            st.write(f"Confidence: {mn_nb_conf:.2f}%")

        with col2:
            st.info(f"Random Forest: {'Fake' if rfc_pred else 'Real'}")
            st.write(f"Confidence: {rfc_conf:.2f}%")

            st.warning(f"Linear SVC: {'Fake' if svc_pred else 'Real'}")
            st.write("Confidence: Not Available")

        # 4️ final verdict
        votes = [lr_pred, mn_nb_pred, rfc_pred, svc_pred]
        final_prediction = max(set(votes), key=votes.count)

        st.subheader(" Final Verdict")

        if final_prediction == 1:
            st.error(" Fake Job Detected (Majority Vote)")
        else:
            st.success(" Real Job Posting (Majority Vote)")
