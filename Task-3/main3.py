import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st

data=pd.read_csv(r"C:\Users\Udhaya kiran\OneDrive\Desktop\oasisinfobyte_tasks\Task-3\modified_car_record.csv")
df=pd.DataFrame(data)
df.drop(df.columns[df.columns.str.contains('Unnamed',case=False)],axis=1,inplace=True)
df['age']= 2025- df['year']
df.drop('year',axis=1,inplace=True)
#print(df.head())
#print(df.columns)
cat_features=['name','fuel','seller_type','transmission','owner',]
num_features=['km_driven','selling_price','mileage','seats']
cat_Transform=OneHotEncoder(handle_unknown='ignore')
preprocess=ColumnTransformer(
  transformers=[('cat',  cat_Transform,cat_features)],
  remainder= 'passthrough'
)
model=RandomForestRegressor(n_estimators=200,
    max_depth=None,
    random_state=42 )
pipe=Pipeline(steps=[
    ('preprocess',preprocess),
    ('model', model)]
)
x=df.drop('selling_price',axis=1)
y=df['selling_price']

print(x.columns)
pipe.fit(x,y)

st.title("CAR PRICE PREDICTION")
st.subheader("FILL UP THE DETAILS")
name=st.selectbox("SELECT BRAD OF THE CAR",df['name'].unique())
km=st.slider("ENTER THE KM DRIVEN" ,1,2360457,1)
fuel=st.selectbox("SELECT FUEL CAPACITY OF THE CAR",df['fuel'].unique())
sel=st.selectbox("SELECT SELLER TYPE",df['seller_type'].unique())
trans=st.selectbox("SELECT PERFERED TRANSMISSION ",df['transmission'].unique())
own=st.selectbox("SELECT THE OWNERSHIP ",df['owner'].unique())
ml=st.slider("ENTER THE MILAGE" ,0.0,10.0)
en=st.slider("ENTER THE ENGINE TYPE" ,310.0,620.0)
max=st.slider("ENTER THE MAXPOWER" ,30.0,400.0)
seat=st.slider("ENTER THE NO OF SEATS" ,2,14)
age=st.slider("ENTER THE AGE OF THECAR" ,5,31,5)
input_df = pd.DataFrame([{
    'name': name,
    'km_driven': km,
    'fuel': fuel,
    'seller_type': sel,
    'transmission': trans,
    'owner': own,
    'mileage': ml,
    'engine': en,
    'max_power': max,
    'seats': seat,
    'age': age
}])


if st.button("PREDICTION"):
    price=pipe.predict(input_df)
    st.write(f"THE PRICE IS {int(price[0]):,}")





