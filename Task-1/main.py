#neccessarry inputs
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
#data loading
data=pd.read_csv(r"C:\Users\Udhaya kiran\OneDrive\Desktop\oasisinfobyte_tasks\Task-1\Iris.csv")
df=pd.DataFrame(data)
#Data precrocessing
le=LabelEncoder()
df.drop('Id',axis=1,inplace=True)
df['Species']=le.fit_transform(df['Species'])
x=df.drop('Species',axis=1)
y=df['Species']
#model defination
model=LogisticRegression(max_iter=200)
model.fit(x,y)
y_pred=model.predict(x)
sns.set_style('darkgrid')
pairplot=sns.pairplot(df)
#STREAMLIT_UI
st.title("IRIS DATASET PREDICTION")
st.dataframe(df.head())
st.dataframe(df.describe())
st.dataframe(df.isnull())
st.write(accuracy_score(y,y_pred))
st.pyplot(pairplot)
st.subheader("ENTER THE LENGHT OF FLOWERS")
sl=st.number_input("sepal_lenght", min_value=1.0, max_value=10.0 ,step=0.5)
sw=st.number_input("sepal_width", min_value=1.0, max_value=10.0 ,step=0.5)
pl=st.number_input("petal_lenght", min_value=1.0, max_value=10.0 ,step=0.5)
pw=st.number_input("petal_width", min_value=1.0, max_value=10.0 ,step=0.5)
input_array=np.array([[sl,sw,pl,pw]])
if st.button("PREDICTION"):
     prediction=model.predict(input_array)
     result=le.inverse_transform(prediction)[0]
     st.write(result)
     
    
