import os
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.sidebar.image(os.path.join('Datasciencepro.png'),None,100)
st.write("""
# HR Analytics - Predict the **Salary** based on **Experience** 
Simple Linear Regression!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Experience = st.sidebar.slider('Experience in Years', 0, 20, 0)
    data = {'Experience': Experience}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# os.chdir(r'C:\Users\Rahul\Desktop\Streamlit')
dataset  = pd.read_csv(os.path.join('Salary_Data.csv'))
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

regressor = LinearRegression()
regressor.fit(X, Y)

prediction = regressor.predict(df)
#prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('The regression coefficients are:')
st.write('Slope of a line:', regressor.coef_)

st.write('Y - Intercept of a line:', regressor.intercept_)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)