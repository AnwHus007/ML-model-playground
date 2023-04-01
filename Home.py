import streamlit as st

st.set_page_config(
    page_title="ML playground app"
    )

st.title("ML playground app")

st.write("Welcome to ML Playground app! This platform is designed to provide you with a user-friendly interface for selecting and evaluating the best machine learning models for regression and classification tasks.")
st.write("This app features a range of algorithms and models that you can experiment with, including linear regression, decision trees, random forests and many more. ")
st.write("Whether you're a beginner or an experienced data scientist,this app can help you quickly and easily compare the performance of different models on your datasets.")
st.write("With this app, you can explore the nuances of different algorithms and you can download the model that best fit your data")

st.image("https://static.javatpoint.com/tutorial/machine-learning/images/regression-vs-classification-in-machine-learning.png")
st.sidebar.success("Select the model")