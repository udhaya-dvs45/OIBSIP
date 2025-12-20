import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv(r"C:\Users\Udhaya kiran\OneDrive\Desktop\oasisinfobyte_tasks\Task-5\Advertising_rounded_500.csv")
df=pd.DataFrame(data)
'''
print(df.agg(['min', 'max']))
print(df.describe())
print(df.isnull().sum())
print(df.duplicated())
print(df.shape)
print(df.info())'''
x=df.drop('Sales',axis=1)
y=df['Sales']
fig1, ax1 = plt.subplots()
ax1.bar(x.columns, y.mean())
ax1.set_title("Average Sales by Channel")
fig2, ax2 = plt.subplots()
ax2.scatter(df['TV'], df['Sales'])
ax2.set_xlabel("TV Advertising Spend")
ax2.set_ylabel("Sales")
ax2.set_title("TV vs Sales")

fig3, ax3 = plt.subplots()
ax3.scatter(df['Newspaper'], df['Sales'])
ax3.set_xlabel("Newspaper Advertising Spend")
ax3.set_ylabel("Sales")
ax3.set_title("Newspaper vs Sales")

fig4, ax4 = plt.subplots()
ax4.scatter(df['Radio'], df['Sales'])
ax4.set_xlabel("Radio Advertising Spend")
ax4.set_ylabel("Sales")
ax4.set_title("Radio vs Sales")

pairplot = sns.pairplot(df)

fig6, ax6 = plt.subplots()
ax6.hist(df['Sales'], bins=10)
ax6.set_xlabel("Sales")
ax6.set_ylabel("Frequency")
ax6.set_title("Distribution of Sales")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression(fit_intercept=True)
model.fit(x_train,y_train)

st.title("SALES PREDICTION ON ADVERTISEMET")
plot_type=st.selectbox("SLELCT THE RELATIOSHIP YOU WANT TO VIEW ",
                 ("Select the graph","Average Sales by Channel","TV vs Sales","Newspaper vs Sales","Radio vs Sales","Distribution of Sales","All Relation"))
if plot_type != "Select a plot":
    fig, ax = plt.subplots()
    if plot_type == "Average Sales by Channel":
        st.pyplot(fig1)

    elif plot_type == "TV vs Sales":
        st.pyplot(fig2)
       
    elif plot_type == "Radio vs Sales":
        st.pyplot(fig4)

    elif plot_type == "Newspaper vs Sales":
        st.pyplot(fig3)

    elif plot_type == "Distribution of Sales":
        st.pyplot(fig6)
    elif plot_type ==  "All Relation":
        st.pyplot(pairplot.fig)
Tv=st.number_input(" ADVERTISEMENT BUDGET ",min_value=0.0, max_value=320.0,step=10.0)
radio=st.number_input("RADIO ADVERTISEMENT BUDGET",min_value=0.0, max_value=65.0,step=1.0)
News=st.number_input("NEWSPAPER ADVERSTISEMET BUDGET",min_value=0.0, max_value=35.0,step=0.5)
if st.button("PREDICTION"):
    input=np.array([[Tv,radio,News]])
    prediction=model.predict(input)
    st.write(f" THE PREDICTED PRICE IS {prediction[0]:.2f}")
    st.image(r"C:\Users\Udhaya kiran\OneDrive\Desktop\oasisinfobyte_tasks\Task-5\sales.png",width=300)