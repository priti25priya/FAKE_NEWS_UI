import streamlit as st
import pickle

# Load trained model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Fake News Prediction App")
st.write("Enter a news article below to check if it's Fake or Real.")

# User Input
user_input = st.text_area("Enter News Article Here:")

# Prediction Button
if st.button("Predict"):
    if user_input:
        # Convert input to TF-IDF format
        input_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vector)

        # Display result
        result = "Fake News ❌" if prediction[0] == 1 else "Real News ✅"
        st.write(f"### Prediction: {result}")
    else:
        st.warning("⚠️ Please enter a news article before predicting.")
