def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)



import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import sklearn

df = pd.read_csv('crop_production.csv')
df2 = pd.read_csv('crop_yield.csv')
grouped = df.groupby('State_Name')['District_Name']
d = {}
for name, group in grouped:
    d[name] = list(np.unique(group.values))
crop_yield_col1 = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']
crop_yield_model1 = joblib.load('crop_yield_pipe1.joblib')

crop_yield_col2 = ['crop_year', 'season_names', 'crop_names', 'area', 'temperature',
       'wind_speed', 'pressure', 'humidity', 'soil_type', 'N', 'P', 'K']
crop_yield_model2 = joblib.load('crop_yield_pipe2.joblib')

crop_model = joblib.load('crop_pred_pipe.joblib')
crop_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
cat_num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fert_model = joblib.load('fert_pred_pipe.joblib')
fert_col = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type','Nitrogen', 'Potassium', 'Phosphorous']
fert_cat_col = ['Soil Type', 'Crop Type']
fert_num_col = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

def fert_prediction(data):
    df = pd.DataFrame(data=[data],columns=fert_col)
    pred = fert_model.predict(df)
    return pred[0]

def crop_prediction(data,k=5):
    df = pd.DataFrame(data=[data],columns=crop_col)
    d = dict(zip(crop_model[1].classes_,crop_model.predict_proba(df).ravel()))
    d = {i:j for i,j in sorted(d.items(),key=lambda x:x[1],reverse=True)}
    return ', '.join(list(d.keys())[:k])

def crop_yield_pred1(data):
    df = pd.DataFrame(data=[data],columns=crop_yield_col1)
    pred = crop_yield_model1.predict(df)
    return pred[0]

def crop_yield_pred2(data):
    df = pd.DataFrame(data=[data],columns=crop_yield_col2)
    pred = crop_yield_model2.predict(df)
    return pred[0]
st.set_page_config(
        page_title="Crop Yield Prediction",
        page_icon='crop.jpg'
        
)   
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.sidebar.title("பயிர் விளைச்சல் கணிப்பு மென்பொருள்")
choice = st.sidebar.radio(label = 'விருப்பத்தைத் தேர்ந்தெடுக்கவும்',options = ['பயிர் விளைச்சல் கணிப்பு','பயிர் பரிந்துரை','உரங்களின் கணிப்பு'])
if choice == 'பயிர் விளைச்சல் கணிப்பு':
    crop_season = np.unique(list(df['Season'].unique()) + list(df2['season_names'].unique()))
    all_crop_names = np.unique(list(df['Crop'].unique()) + list(df2['crop_names'].unique()))
    st.title('பயிர் விளைச்சல் முன்னறிவிப்பு மாதிரி')
    st.image('crop_yield.jpg')
    st.text('கீழே கொடுக்கப்பட்டுள்ள விவரங்கள் அந்த பகுதியில் பயிர் விளைச்சலைக் கணிக்கின்றன')
    states = list(d.keys())
    state = st.selectbox(label="மாவட்டத்தை தேர்வு செய்யவும்",options=states)
    district = st.selectbox(label="மாநிலத்தை தேர்வு செய்யவும்",options=d[state])
    Crop_Year = st.number_input(label="வருடத்தை உள்ளிடவு",min_value=2000)
    Season = st.selectbox(label="பருவத்தைத் தேர்ந்தெடுக்கவும்",options=crop_season)
    crop = st.selectbox(label="பயிர் வகையைத் தேர்ந்தெடுக்கவும்",options=all_crop_names)
    area = float(st.number_input(label="ஏக்கரில் ஏரியா அளவை உள்ளிடவும்",min_value=0))
    temp = st.number_input(label="டிகிரி செல்சியஸில் வெப்பநிலையை உள்ளிடவும்",step=1.,format="%.2f")
    wind_speed = st.number_input(label="காற்றின் வேக மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    pressure = st.number_input(label="அழுத்த மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    humidity = st.number_input(label="ஈரப்பதத்தின் மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    soil_type = st.selectbox(label="மண் வகையைத் தேர்ந்தெடுக்கவும்",options=df2['soil_type'].unique())
    N = float(st.number_input(label="நைட்ரஜன் மதிப்பை உள்ளிடவும்"))
    P = float(st.number_input(label="பாஸ்பரஸ் மதிப்பை உள்ளிடவும்"))
    K = float(st.number_input(label="பொட்டாசியம் மதிப்பை உள்ளிடவும்"))

    predict = st.button("கணிக்கவும்")
    if predict:
        data = [state,district,Crop_Year,Season,crop,area]
        pred1 = crop_yield_pred1(data)
        data2 = [Crop_Year,Season,crop,area,temp,wind_speed,pressure,humidity,soil_type,N,P,K]
        pred2 = crop_yield_pred2(data2)
        pres='கொடுக்கப்பட்ட உள்ளீட்டு அளவுருக்கள் மற்றும் பகுதிக்கான கணிக்கப்பட்ட பயிர் விளைச்சல்\n'
        ton=' டன்கள்'
        ans=pres + str((pred1+pred2)/2) + ton
        st.subheader(ans)
 
        

elif choice == 'பயிர் பரிந்துரை':
    st.title('பயிர் பரிந்துரை மாதிரி')
    st.image('crop.jpg')
    st.text('கீழே கொடுக்கப்பட்டுள்ள விவரங்கள் இப்பகுதிக்கு ஏற்ற சிறந்த பயிர்களை கணிக்கின்றன')
    states = list(d.keys())
    state = st.selectbox(label="மாநிலத்தை தேர்வு செய்யவும்",options=states)
    district = st.selectbox(label="மாவட்டத்தை தேர்வு செய்யவும்",options=d[state])
    N = st.number_input(label="நைட்ரஜன் மதிப்பை உள்ளிடவும்")
    P = st.number_input(label="பாஸ்பரஸ் மதிப்பை உள்ளிடவும்")
    K = st.number_input(label="பொட்டாசியம் மதிப்பை உள்ளிடவும்")
    temp = st.number_input(label="டிகிரி செல்சியஸில் வெப்பநிலையை உள்ளிடவும்",step=1.,format="%.2f")
    hum = st.number_input(label="ஈரப்பதத்தின் மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    ph = st.number_input(label="ph மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    rainfall = st.number_input(label="மில்லிமீட்டர் மழை மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    predict = st.button("கணிக்கவும்")
    if predict:
        data = [N,P,K,temp,hum,ph,rainfall]
        data = [float(i) for i in data]
        pred = crop_prediction(data)
        st.subheader(f"கொடுக்கப்பட்ட நிபந்தனைகளுக்கு பரிந்துரைக்கப்பட்ட பயிர்கள்:\n {pred}")
else:
    st.title('உர கணிப்பு மாதிரி')
    st.image('fertilizers.jpg')
    st.text('கீழே கொடுக்கப்பட்டுள்ள விவரங்கள் பயிருக்கு சிறந்த உரத்தை கணிக்கின்றன')
    states = list(d.keys())
    state = st.selectbox(label="மாநிலத்தை தேர்வு செய்யவும்",options=states)
    district = st.selectbox(label="மாவட்டத்தை தேர்வு செய்யவும்",options=d[state])
    temp = st.number_input(label="டிகிரி செல்சியஸில் வெப்பநிலையை உள்ளிடவும்",step=1.,format="%.2f")
    hum = st.number_input(label="ஈரப்பதத்தின் மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    moist = st.number_input(label="ஈரப்பத மதிப்பை உள்ளிடவும்",step=1.,format="%.2f")
    c1 = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    soil_type = st.selectbox(label="மண் வகையைத் தேர்ந்தெடுக்கவும்",options=c1)
    c2 = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
        'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    crop_type = st.selectbox(label="பயிர் வகையைத் தேர்ந்தெடுக்கவும்",options=c2)
    N = st.number_input(label="நைட்ரஜன் மதிப்பை உள்ளிடவும்")
    Pt = st.number_input(label="பொட்டாசியம் மதிப்பை உள்ளிடவும்")
    Phs = st.number_input(label="பாஸ்பரஸ் மதிப்பை உள்ளிடவும்")
    predict = st.button("கணிக்கவும்")
    if predict:
        data = [temp,hum,moist,soil_type,crop_type,N,Pt,Phs]
        pred = fert_prediction(data)
        st.subheader(f"கொடுக்கப்பட்ட நிபந்தனைக்கான உரங்களுக்கான பரிந்துரைக்கப்பட்ட NPK உள்ளடக்கம்:\n {pred}")
