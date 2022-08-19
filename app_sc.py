from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# load trained classifier

#X_test = pd.read_csv('application_test.csv', index_col='SK_ID_CURR', encoding ='utf-8')

def load_data():
    
    X_test = pd.read_csv('application_test1.csv', index_col='SK_ID_CURR', sep = ';',  encoding ='utf-8')
    
    columns_explicative = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
                      'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'AMT_ANNUITY',
                      'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT']
    
    X_test = X_test[columns_explicative]
    
    # --- processing 

    # imputer for handling missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Fit on the training data
    imputer.fit(X_test[columns_explicative])

    # Transform both training and testing data
    X_test[columns_explicative] = imputer.transform(X_test[columns_explicative])

    # Repeat with the scaler
    scaler.fit(X_test[columns_explicative])
    X_test[columns_explicative] = scaler.transform(X_test[columns_explicative])
    
    return X_test

X_test1=load_data()


clf_path = 'model_vf.pkl'

with open(clf_path, 'rb') as f:
    model = pickle.load(f) 
    
# #importer le modèle
# load_model=pickle.load(open('prevision_credit_rand_forest.pkl','rb'))
    
#{"SK_ID_CURR": 12333}

@app.route('/api',methods=['POST'])

def predict():

    data = request.get_json(force=True)

    SK_ID_CURR=data['SK_ID_CURR']  

    client = X_test1.loc[X_test1.index == int(SK_ID_CURR)]
    
    prediction = model.predict(client)
    probability= model.predict_proba(client)

    output={'prediction': str(prediction[0]),'probability': probability.max()}

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    