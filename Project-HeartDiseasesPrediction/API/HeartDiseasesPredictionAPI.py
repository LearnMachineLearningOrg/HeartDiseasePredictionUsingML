#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request

#library for saving the trained models to files
import joblib

import pandas as pd

#Libraries for feature encoding
from sklearn.preprocessing import LabelEncoder

#feature encoding library
from featureencodinglibrary import featureEncodingUsingOneHotEncoder

#feature scaling library
from featurescalinglibrary import featureScalingUsingNormalizer


# In[2]:


app = Flask(__name__)      


# In[3]:


@app.route('/predictHeartDiseaseUsingML', methods=['POST'])
def predictHeartDiseaseUsingML():  
    # use parser and find the user's query
    jsonRequest = request.get_json()
    
    data = [jsonRequest]
    testDataFrame = pd.DataFrame(data)
    
    #Perform the same preprocessing as performed on training dataset
    testDataFrameEncoded = featureEncodingUsingOneHotEncoder(testDataFrame)
    testDataFrameEncodedAndScaledDataset = featureScalingUsingNormalizer(testDataFrameEncoded)

    #Load classifier from file
    xtest = testDataFrameEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testDataFrameEncodedAndScaledDataset.iloc[:, len(testDataFrameEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)
    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    prediction = {'prediction':int(ytestpred[0])}
    return jsonify(prediction)
    #return '''ytestpred: {}'''.format(ytestpred)


# In[4]:


if __name__ == '__main__':
     classifier = joblib.load('OneHotEncoder_Normalizing_RandomForestClassifier.pkl')
     app.run(port=8080)


# In[ ]:




