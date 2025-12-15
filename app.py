import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_icon="💰", page_title="Income Predictor")
st.title("Предскажи свой доход")
st.subheader("без регистрации и смс")
st.markdown("---")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

model, feature_names = load_model()

# сайдбар с вводом
st.sidebar.header("Введите ваши данные")
age = st.sidebar.slider("Возраст", 17, 90, 22)
workclass = st.sidebar.selectbox("Тип занятости", ['Private', 'Self-emp-not-inc', 'Local-gov'])
education = st.sidebar.selectbox("Образование", ['Bachelors', 'HS-grad', 'Some-college'])
marital_status = st.sidebar.selectbox("Семья", ['Married-civ-spouse', 'Never-married'])
occupation = st.sidebar.selectbox("Профессия", ['Prof-specialty', 'Craft', 'Exec-managerial'])
relationship = st.sidebar.selectbox("Семейное положение", ['Husband', 'Not-in-family', 'Wife'])
race = st.sidebar.selectbox("Раса", ['White', 'Black'])
sex = st.sidebar.selectbox("Пол", ['Male', 'Female'])
capital_gain = st.sidebar.slider("Прибыль", 0, 100000, 0)
capital_loss = st.sidebar.slider("Убытки", 0, 5000, 0)
hours_per_week = st.sidebar.slider("Занятость (часов/неделю)", 1, 99, 40)

if st.sidebar.button("Предсказать", type="primary"):
    # Создаем данные
    input_data = {
        'age': age, 'workclass': workclass, 'education': education,
        'marital-status': marital_status, 'occupation': occupation,
        'relationship': relationship, 'race': race, 'sex': sex,
        'capital-gain': capital_gain, 'capital-loss': capital_loss,
        'hours-per-week': hours_per_week
    }
    
    input_df = pd.DataFrame([input_data])
    
    # One-hot encoding
    input_cat = pd.get_dummies(input_df[['workclass', 'education', 'marital-status', 
                                        'occupation', 'relationship', 'race', 'sex']])
    
    # Объединяем числовые + категориальные
    input_num = input_df[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
    input_full = pd.concat([input_num, input_cat], axis=1)
    
    # Подгоняем под обученные признаки
    input_full = input_full.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Предсказание
    prob = model.predict_proba(input_full)[0][1]
    prediction = "✅ > $50K" if prob > 0.5 else "❌ ≤ $50K"
    
    # Результат
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("С вероятностью", f"{prob:.1%}")
    with col2:
        st.success(f"**{prediction}**")
