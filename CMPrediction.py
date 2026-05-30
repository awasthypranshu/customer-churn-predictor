import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

with open("gender_encoder.pkl","rb") as file:
    gender_encoder = pickle.load(file)

with open("geo_ohe.pkl","rb") as file:
    geo_ohe = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model("model.h5")

# input_data = {
#     'CreditScore': 600,
#     'Geography': 'France',
#     'Gender': 'Male',
#     'Age': 40,
#     'Tenure': 3,
#     'Balance': 60000,
#     'NumOfProducts': 2,
#     'HasCrCard': 1,
#     'IsActiveMember': 1,
#     'EstimatedSalary': 50000
# }

# input_df = pd.DataFrame([input_data])

# input_df["Gender"] = gender_encoder.transform(input_df['Gender'])

# geography_encoded = geo_ohe.transform([input_df["Geography"]]).toarray()
# geo_enc_df = pd.DataFrame(geography_encoded,columns=geo_ohe.get_feature_names_out())

# input_df = pd.concat([input_df.drop("Geography",axis=1),geo_enc_df],axis=1)

# scaled_data = scaler.transform(input_df)

# prediction = model.predict(scaled_data)
# print(prediction)



# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox(
    'Geography',
    geo_ohe.categories_[0]
)
gender = st.selectbox(
    'Gender',
    gender_encoder.classes_
)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore':[credit_score],
    'Geography': [geography],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}


data = pd.DataFrame(input_data)
encoded_geography = geo_ohe.transform(data[["Geography"]]).toarray()
geo_df = pd.DataFrame(encoded_geography,columns=geo_ohe.get_feature_names_out())

data = pd.concat([data.drop("Geography",axis=1),geo_df],axis=1)

scaled_data = scaler.transform(data)
prediction = model.predict(scaled_data)

if prediction[0][0] > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is likely to NOT churn")    