import pandas as pd
import numpy as np
import streamlit as st
import skops.io as sio
from pickle import dump, load
import os

## Load model, encoder, and explainer
unknown_types = sio.get_untrusted_types(file="data/modelling/model.skops")
model = sio.load("data/modelling/model.skops", trusted=unknown_types)

unknown_types = sio.get_untrusted_types(file="data/feature_eng/encoder.skops")
encoder = sio.load("data/feature_eng/encoder.skops", trusted=unknown_types)

## define log proba to be used for explainer
def predict_log_proba(z):
    p = model.predict_proba(z)
    return np.log(p[:,1] / p[:,0])

with open('data/explainer/explainer.pkl', 'rb') as file:
    explainer = load(file)

## define encoder transform
def enc_transform(X, enc):
    cat_feat = enc.get_feature_names_out()
    X_ord = X[cat_feat]
    X_enc = enc.transform(X_ord)
    X_enc = pd.DataFrame(X_enc, index=X.index,
                         columns=enc.get_feature_names_out())
    
    X[cat_feat] = X_enc
    return X

st.write("""
         # Simple Customer Churn Prediction
         **by:** [Muhammad Robith](https://www.linkedin.com/in/mrobith95) \n
         This web-app is a part of project for [pacmann](https://pacmann.io)'s ML APIs & Deployment Course. \n
         Dataset considered for this web-app available [here](https://www.kaggle.com/datasets/willianoliveiragibin/customer-churn). \n
         **DISCLAIMER:** This web-app is merely for portfolio show-off only, and not related to any company. \n
         ## Customer Info
         Enter Customer Information Here \n
         """)

credit_score = st.number_input(label='**Credit Score**',
                               min_value=0,
                               value=714)

if credit_score < 350:
    st.toast(f"Credit score ({credit_score}) is lower than dataset (350), thus expect unexpected result")
elif credit_score > 850:
    st.toast(f"Credit score ({credit_score}) is higher than dataset (850), thus expect unexpected result")

country = st.selectbox(label="**Country**",
                       options=('France', 'Germany', 'Spain', 'Other'))

gender = st.selectbox(label="**Gender**",
                      options=('Female', 'Male', 'Other'))

age = st.number_input(label='**Age**',
                      min_value=0,
                      value=53)

if age < 18:
    st.toast(f"Age value ({age}) is lower than dataset (18), thus expect unexpected result")
elif age > 92:
    st.toast(f"Age value ({age}) is higher than dataset (92), thus expect unexpected result")

tenure = st.number_input(label='**Tenure**',
                         min_value=0,
                         value=1)

if tenure > 10:
    st.toast(f"Tenure ({tenure}) is higher than dataset (10), thus expect unexpected result")

balance = st.number_input(label='**Balance**',
                          min_value=0.0,
                          value=99141.86)

if balance > 250898.09:
    st.toast(f"Balance ({balance}) is higher than dataset (250898.09), thus expect unexpected result")

salary = st.number_input(label='**Estimated Salary**',
                         min_value=0.0,
                         value=72496.05)

if salary < 11.58:
    st.toast(f"Estimated Salary ({salary}) is lower than dataset (11.58), thus expect unexpected result")
elif salary > 199992.48:
    st.toast(f"Estimated Salary ({salary}) is higher than dataset (199992.48), thus expect unexpected result")

if st.button(label="**Predict!**", type='primary'):
    ini_dict = {
        'CreditScore': credit_score,
        'Geography': country,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'EstimatedSalary': salary
    }
    ini_df = pd.DataFrame(ini_dict, index=[0])
    ini_df = enc_transform(ini_df, encoder)
    pred = model.predict_proba(ini_df)
    predv = 100*round(float(pred[0][1]),4)
    if pred[0][1]>=0.5:
        pred_text = 'Churn'
    else:
        pred_text = 'Not Churn'
    shap_values = explainer(ini_df)
    shap_dict = {
        'CreditScore': float(shap_values.values[0][0]),
        'Geography': float(shap_values.values[0][1]),
        'Gender': float(shap_values.values[0][2]),
        'Age': float(shap_values.values[0][3]),
        'Tenure': float(shap_values.values[0][4]),
        'Balance': float(shap_values.values[0][5]),
        'EstimatedSalary': float(shap_values.values[0][6])
    }
    sorted_keys = sorted(shap_dict, key=lambda x: shap_dict[x])
    non_churn_reason = ' '
    if shap_dict[sorted_keys[0]]<0:
        non_churn_reason += sorted_keys[0]+' ('+str(round(shap_dict[sorted_keys[0]],2))+') '
    if shap_dict[sorted_keys[1]]<0:
        non_churn_reason += '& '+sorted_keys[1]+' ('+str(round(shap_dict[sorted_keys[1]],2))+') '
    churn_reason = ' '
    if shap_dict[sorted_keys[6]]>0:
        churn_reason += sorted_keys[6]+' ('+str(round(shap_dict[sorted_keys[6]],2))+') '
    if shap_dict[sorted_keys[5]]>0:
        churn_reason += '& '+sorted_keys[5]+' ('+str(round(shap_dict[sorted_keys[5]],2))+') '
else:
    pred_text = '-'
    non_churn_reason = '-'
    churn_reason = '-'
    predv = '-'

st.write("""
         ## Prediction
         """)

st.markdown("**Prediction:** "+pred_text)
st.markdown("Churn probability: "+str(predv)+"%")
st.markdown("Features that supports churn:"+churn_reason)
st.markdown("Features that against churn:"+non_churn_reason)

# st.text_input(label='**Prediction**',
#               value=pred_text,
#               disabled=True)