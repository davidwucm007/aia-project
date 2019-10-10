#!/usr/bin/env python
# coding: utf-8

# In[186]:


#Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[187]:


# Load data
data = pd.read_csv('adult_data.csv', index_col=False,delimiter=", ")
data.head()


# In[188]:


#Check for null values and datatypes
print(data.isnull().sum())
print(data.dtypes)


# In[189]:


#check ' ?' containing rows
def cc(x):
    return sum(x=='?')
data.apply(cc)


# In[190]:


data.shape


# In[191]:


#Label the income objects as 0 and 1
data['income']=data['income'].map({'<=50K': 0, '>50K': 1})


# In[192]:


#Drop "education" as the columns "education" and "education-num" convey the same meaning
#Drop "relationship" as "marital-status" and "relationship" convey the same meaning as well
#Drop "occupation" as "workclass" is sufficient enough
#Drop "fnlwgt" as it is not useful
#Drop "native.country" as most are from the US, which causes bias
data=data.drop(['education', 'relationship', 'occupation', 'fnlwgt', 'native-country'], axis=1)


# In[193]:


#Reformat marital-status values to single and married
data['marital-status']=data['marital-status'].map({'Married-civ-spouse':'Married', 'Divorced':'Single', 'Never-married':'Single', 'Separated':'Single', 
'Widowed':'Single', 'Married-spouse-absent':'Married', 'Married-AF-spouse':'Married'})


# In[194]:


#Separate target from predictors
y = data.income
X = data.drop(['income'], axis=1)


# In[195]:


#Select categorical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
categorical_cols


# In[196]:


#Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values='?', strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols)
])


# In[197]:


#Try out logistic regression, decision tree, random forest, xgboost



#1. Logistic regression
from sklearn.linear_model import LogisticRegression


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LogisticRegression())
                             ])


from sklearn.model_selection import cross_val_score

#Calculates accuracy
scores = cross_val_score(my_pipeline, X, y,
                              cv=5)

print("Accuracy scores:\n", scores)


# In[198]:


print("Average accuracy score (across experiments):")
print(scores.mean()*100)


# In[199]:


# Prepare test data
data_test = pd.read_csv('adult_test.csv', index_col=False,delimiter=", ")
data_test['income']=data_test['income'].map({'<=50K.': 0, '>50K.': 1})
data_test=data_test.drop(['education', 'relationship', 'occupation', 'fnlwgt', 'native-country'], axis=1)
data_test['marital-status']=data_test['marital-status'].map({'Married-civ-spouse':'Married', 'Divorced':'Single', 'Never-married':'Single', 'Separated':'Single', 
'Widowed':'Single', 'Married-spouse-absent':'Married', 'Married-AF-spouse':'Married'})
y_test = data_test.income
X_test = data_test.drop(['income'], axis=1)


# In[200]:


#Preprocessing of all training data, fit model 
my_pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = accuracy_score(y_test, preds)
print('Accuracy:', score*100)


# In[201]:


#2. Decision tree
from sklearn.tree import DecisionTreeClassifier


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', DecisionTreeClassifier())
                             ])



#Calculates accuracy
scores = cross_val_score(my_pipeline, X, y,
                              cv=5)

print("Accuracy scores:\n", scores)


# In[202]:


print("Average accuracy score (across experiments):")
print(scores.mean()*100)


# In[203]:


#Preprocessing of all training data, fit model 
my_pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = accuracy_score(y_test, preds)
print('Accuracy:', score*100)


# In[204]:


#3. Random forest
from sklearn.ensemble import RandomForestClassifier


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier())
                             ])



#Calculates accuracy
scores = cross_val_score(my_pipeline, X, y,
                              cv=5)

print("Accuracy scores:\n", scores)


# In[205]:


print("Average accuracy score (across experiments):")
print(scores.mean()*100)


# In[206]:


#Preprocessing of all training data, fit model 
my_pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = accuracy_score(y_test, preds)
print('Accuracy:', score*100)


# In[207]:


#4. xgboost
from xgboost import XGBClassifier


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', XGBClassifier(n_estimators=500,learning_rate=0.05, early_stopping_rounds=5))
                             ])



#Calculates accuracy
scores = cross_val_score(my_pipeline, X, y,
                              cv=5)

print("Accuracy scores:\n", scores)


# In[208]:


print("Average accuracy score (across experiments):")
print(scores.mean()*100)


# In[209]:


#Preprocessing of all training data, fit model 
my_pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = accuracy_score(y_test, preds)
print('Accuracy:', score*100)


# In[ ]:




