import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# stop words used such as : i me you will would etc
nltk.download('stopwords')

# Loading model & Vectorizer
lr_model=pickle.load(open('lr_model.pkl','rb'))

vectorizer=pickle.load(open('vectorizer.pkl','rb'))

# port stemmer initialization
stem_port=PorterStemmer()

def clean_text(text):
    text=re.sub('[^a-zA-Z]', ' ' ,text) # choosing only english characters
    text=text.lower() # cchanging it into the lower case letters
    text=text.split() # splitting text into the list of words
    text=[stem_port.stem(word) for word in text if not word in stopwords.words('english')]
    text= ' ' .join(text)
    return text

# UI For APP
st.set_page_config(page_title='Fake Job Detector',layout='centered')

# title
st.title('Fake job Posting Detector')

# write
st.write('Paste job description below to check authenticity')

# user input about the job
user_input=st.text_area('Job Description',height=200)

if st.button('Check Job'):
    if user_input.strip()=='':
        st.warning('Please enter job description')
    else:
        cleaned=clean_text(user_input)
        vectorized=vectorizer.transform([cleaned])
        prediction=lr_model.predict(vectorized)[0]
        probability=lr_model.predict_proba(vectorized)[0]


        if prediction==0:
            st.success(f'Real Job Posting (Probability: {probability[0]*100:.2f}%)')
        else:
            st.error(f'Fake Job Detected (Probability: {probability[1]*100:.2f}%)')

