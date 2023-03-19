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
st.sidebar.title("Crop Yield Prediction Application")
choice = st.sidebar.radio(label = 'Select option',options = ['Crop Yield Prediction','Crop Recommendation','Fertilizers Prediction'])
if choice == 'Crop Yield Prediction':
    crop_season = np.unique(list(df['Season'].unique()) + list(df2['season_names'].unique()))
    all_crop_names = np.unique(list(df['Crop'].unique()) + list(df2['crop_names'].unique()))
    st.title('Crop Yield Prediction Model')
    st.image('crop_yield.jpg')
    st.text('Given below details it predicts crop yield in that area')
    states = list(d.keys())
    state = st.selectbox(label="Choose State",options=states)
    district = st.selectbox(label="Choose District",options=d[state])
    Crop_Year = st.number_input(label="Enter Year",min_value=2000)
    Season = st.selectbox(label="Choose Season",options=crop_season)
    crop = st.selectbox(label="Choose Crop",options=all_crop_names)
    area = float(st.number_input(label="Enter Area Size in Acres",min_value=0))
    temp = st.number_input(label="Enter Temperature Value",step=1.,format="%.2f")
    wind_speed = st.number_input(label="Enter Wind Speed value",step=1.,format="%.2f")
    pressure = st.number_input(label="Enter Pressure Value",step=1.,format="%.2f")
    humidity = st.number_input(label="Enter Humidity Value",step=1.,format="%.2f")
    soil_type = st.selectbox(label="Choose Soil Type",options=df2['soil_type'].unique())
    N = float(st.number_input(label="Enter Nitrogen value"))
    P = float(st.number_input(label="Enter Phosphorus value"))
    K = float(st.number_input(label="Enter Potassium value"))

    predict = st.button("Predict")
    if predict:
        data = [state,district,Crop_Year,Season,crop,area]
        pred1 = crop_yield_pred1(data)
        data2 = [Crop_Year,Season,crop,area,temp,wind_speed,pressure,humidity,soil_type,N,P,K]
        pred2 = crop_yield_pred2(data2)
        pres='The Predicted Crop Yield for the given Input Parameters and Area is \n'
        ton=' Tons'
        ans=pres + str((pred1+pred2)/2) + ton
        st.text(ans)
 
        

elif choice == 'Crop Recommendation':
    st.title('Crop Recommendation Model')
    st.image('crop.jpg')
    st.text('Given below details it predicts the best suitable crop for the region')
    states = list(d.keys())
    state = st.selectbox(label="Choose State",options=states)
    district = st.selectbox(label="Choose district",options=d[state])
    N = st.number_input(label="Enter Nitrogen value")
    P = st.number_input(label="Enter Phosphorus value")
    K = st.number_input(label="Enter Potassium value")
    temp = st.number_input(label="Enter Temperature in degree celcius",step=1.,format="%.2f")
    hum = st.number_input(label="Enter humidity value",step=1.,format="%.2f")
    ph = st.number_input(label="Enter ph value",step=1.,format="%.2f")
    rainfall = st.number_input(label="Enter rainfall value in mm",step=1.,format="%.2f")
    predict = st.button("Predict")
    if predict:
        data = [N,P,K,temp,hum,ph,rainfall]
        data = [float(i) for i in data]
        pred = crop_prediction(data)
        st.markdown(f"The Recommended Crops for Given Conditions are:\n {pred}")
else:
    st.title('Fertilizer Prediction Model')
    st.image('fertilizers.jpg')
    st.text('Given below details it predicts the best suitable fertilizer for the crop')
    states = list(d.keys())
    state = st.selectbox(label="Choose State",options=states)
    district = st.selectbox(label="Choose district",options=d[state])
    temp = st.number_input(label="Enter Temperature in degree celcius",step=1.,format="%.2f")
    hum = st.number_input(label="Enter humidity value",step=1.,format="%.2f")
    moist = st.number_input(label="Enter mositure value",step=1.,format="%.2f")
    c1 = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    soil_type = st.selectbox(label="Choose soil-type",options=c1)
    c2 = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
        'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    crop_type = st.selectbox(label="Choose crop-type",options=c2)
    N = st.number_input(label="Enter Nitrogen value")
    Pt = st.number_input(label="Enter Potassium value")
    Phs = st.number_input(label="Enter Phosphorus value")
    predict = st.button("Predict")
    if predict:
        data = [temp,hum,moist,soil_type,crop_type,N,Pt,Phs]
        pred = fert_prediction(data)
        st.text(f"The Recommended NPK content for Fertilizers for the given condition is\n : {pred}")
