import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import lime
from lime import lime_tabular
import streamlit.components.v1 as cp
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# clf_inter = 'model_vf.pkl'
# with open(clf_inter, 'rb') as f:
#     local_inter = pickle.load(f) 
    
#importer le modèle
# load_model=pickle.load(open('model_vf.pkl','rb'))


url='http://kante.pythonanywhere.com/api'

@st.cache
def load_data():
    
    df_test = pd.read_csv('application_test1.csv', index_col='SK_ID_CURR', sep =';' , encoding ='utf-8')
    
    columns_explicative = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
                      'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'AMT_ANNUITY',
                      'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT']
    
    df_test = df_test[columns_explicative]

    # cust_ID = 'SK_ID_CURR'
    

    # --- processing 

    # imputer for handling missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Fit on the training data
    imputer.fit(df_test[columns_explicative])

    # Transform both training and testing data
    df_test[columns_explicative] = imputer.transform(df_test[columns_explicative])

    # Repeat with the scaler
    scaler.fit(df_test[columns_explicative])
    df_test[columns_explicative] = scaler.transform(df_test[columns_explicative])
    
    return df_test

data1=load_data()

#Title
st.title('Dashboard Credit Scoring')
#Infos personnel du client
#st.header("**Information du client**")
#revenu= st.dataframe(data1[data1.index == int(id)])
#age
#sexe
#Family status
#Nb_enfant
#Montant du crédit


  
#interpretor = lime_tabular.LimeTabularExplainer(
 #   training_data=np.array(data1),
 #   feature_names=data1.columns,
 #   mode='classification'
#)

list_ind=list(data1.index)

#sidebar
id=st.sidebar.selectbox('choisir un client',list_ind)

st.dataframe(data1[data1.index == int(id)])

# json
data_json= {'SK_ID_CURR':id}
response=requests.post(url=url,json=data_json).json()
st.write(response)

# proba
if int(response['prediction'])==1:
    st.write('Le client est solvable avec une probabilité de '+str(response['probability']))
else :
    st.write('Le client est non solvable avec une probabilité de '+str(response['probability']))
    
    
    