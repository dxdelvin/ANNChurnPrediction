import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import math

model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)


with open("OneHot_encoder.pkl", "rb") as file:
    OneHot_encoder = pickle.load(file)

    
with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)

#Streamlit app
st.title("Customer Churn Prediction")

#UserInput
geography = st.selectbox("Geography", OneHot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",  18,99)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 1,10)
num_of_products = st.slider("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

#prepare
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age": age,
    "Tenure":tenure,
    "Balance": balance,
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = OneHot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHot_encoder.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scalar.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_proba =  prediction[0][0]

st.write("Churn Probability "+ str(round(prediction_proba,2)))
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')