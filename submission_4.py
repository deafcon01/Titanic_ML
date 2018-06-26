#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:23:15 2018

@author: Rahul
"""

#%%
import numpy as np
import pandas as pd
from sklearn import svm, neighbors, metrics
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
#%%
train_df = pd.read_csv('dataset/train.csv', sep=',')
test_df = pd.read_csv('dataset/test.csv', sep=',')

#%%
train_df.columns[train_df.isnull().any()].tolist()
#%%
test_df.columns[test_df.isnull().any()].tolist()
#%%
train_df['Age'].isnull().value_counts()
#%%
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#%%
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

#%%
train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'), inplace=True)

test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'), inplace=True)

#%%
train_df['Embarked'] = train_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].fillna('S')
#%%
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
#%%
#train_df['Fare']=train_df['Fare'].fillna(train_df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
#%%
#Feature Scaling
# Importing MinMaxScaler and initializing it
min_max=MinMaxScaler()
# Scaling down both train and test data set
train_df[['Fare','Age']]=min_max.fit_transform(train_df[['Fare', 'Age']])

test_df[['Fare','Age']]=min_max.fit_transform(test_df[['Fare', 'Age']])
#%%
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_pass_id = test_df['PassengerId']
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
#%%
#Label encoding
#'Survived','Sex','SibSp','Parch','Embarked'v
labelencoder_sex= LabelEncoder()
train_df['Sex'] = labelencoder_sex.fit_transform(train_df['Sex'])
    
labelencoder_SibSp= LabelEncoder()
train_df['SibSp'] = labelencoder_SibSp.fit_transform(train_df['SibSp'])
    
labelencoder_Parch= LabelEncoder()
train_df['Parch'] = labelencoder_Parch.fit_transform(train_df['Parch'])
    
labelencoder_Embarked= LabelEncoder()
train_df['Embarked'] = labelencoder_Embarked.fit_transform(train_df['Embarked'])

labelencoder_Pclass= LabelEncoder()
train_df['Pclass'] = labelencoder_Pclass.fit_transform(train_df['Pclass'])

labelencoder_Title= LabelEncoder()
train_df['Title'] = labelencoder_Title.fit_transform(train_df['Title'])

test_df['Sex'] = labelencoder_sex.fit_transform(test_df['Sex'])
    
test_df['SibSp'] = labelencoder_SibSp.fit_transform(test_df['SibSp'])
    
test_df['Parch'] = labelencoder_Parch.fit_transform(test_df['Parch'])
    
test_df['Embarked'] = labelencoder_Embarked.fit_transform(test_df['Embarked'])

test_df['Title'] = labelencoder_Title.fit_transform(test_df['Title'])

test_df['Pclass'] = labelencoder_Pclass.fit_transform(test_df['Pclass'])
#%%
Y_label = train_df['Survived']
train_df = train_df.drop(['Survived'], axis=1)
#%%
frames = [train_df, test_df]
DF = pd.concat(frames)

#%%
#One hot encoding
onehotencoder_train = OneHotEncoder(categorical_features = [0,1,3,4,6,7])
DF = onehotencoder_train.fit_transform(DF).toarray()
#%%
#np.random.shuffle(DF)
#%%
DF_train = DF[:891,:]
DF_test = DF[891:,:]
#%%
#One hot encoding
#onehotencoder_train = OneHotEncoder(categorical_features = [0,1,3,4,6,7])
#train_df = onehotencoder_train.fit_transform(train_df).toarray()
#%%
#onehotencoder_test = OneHotEncoder(categorical_features = [0,1,3,4,6,7])
#test_df = onehotencoder_test.fit_transform(test_df).toarray()

#%%
#splitt dataset
X_train, X_test, y_train, y_test = train_test_split(DF_train,Y_label,test_size=0.2, random_state=0)

#%%
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred = logistic.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pred)*100
print("accuracy: ", acc)

#test_data_pred = logistic.predict(test_data)

#%%
#KNN
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
match = 0
acc=metrics.accuracy_score(y_test,y_pred)*100
print("accuracy: ", acc)

#%%
#SVM
model = svm.SVC(kernel='rbf', C=1000, gamma=0.0001) 
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pred)*100
print("accuracy: ", acc)
#%%
#MLP
model=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(10,10),learning_rate_init=1,random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pred)*100
print("accuracy: ", acc)

#%%
# Random Forest


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
#%%
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=25, warm_start=True)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
model.score(X_train, y_train)
acc_model = round(model.score(X_train, y_train) * 100, 2)
acc_model
#%%
acc=[]
kf=KFold(n_splits=10)
i=0
#model = DecisionTreeClassifier()
#model=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(10,10),learning_rate_init=1,random_state=0)
#model = RandomForestClassifier(n_estimators=100)
#model = svm.SVC(kernel='linear', C=1, gamma='auto') 
model = GradientBoostingClassifier(n_estimators=23, warm_start=True)

for train_index, test_index in kf.split(DF_train):
    X_train, X_test = DF_train[train_index], DF_train[test_index]
    y_train, y_test = Y_label[train_index], Y_label[test_index]
    
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=acc+[metrics.accuracy_score(y_test,y_pred)*100]

acc=np.array(acc)
print(acc)
np.mean(acc)
#%%
test_data_pred_RF = model.predict(DF_test)
#%%
submission = pd.DataFrame({
        "PassengerId": test_pass_id,
        "Survived": test_data_pred_RF
    })
submission.to_csv('GradientBoostingClassifier_with_kfold.csv', index=False)
#%%