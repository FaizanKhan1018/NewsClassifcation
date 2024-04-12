import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Main content
st.title('Text Classification with Random Forest')

# Text input for prediction
text_input = st.text_area('Enter text for prediction:', '')

if st.button('Predict'):
    if text_input:
        # Transform input text
        text_transformed = vectorizer.transform([text_input])

        # Make prediction
        prediction = model.predict(text_transformed)
        st.success(f'Prediction: {prediction[0]}')
    else:
        st.warning('Please enter text for prediction.')
