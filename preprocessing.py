import pandas as pd
import matplotlib.pyplot as plt
X_train=pd.read_csv('X_train.csv')
Y_train=pd.read_csv('Y_train.csv')
X_test=pd.read_csv('X_test.csv')
Y_test=pd.read_csv('Y_test.csv')
print (X_train.head())
# Initializing and Fitting a k-NN model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount',
                   'Loan_Amount_Term', 'Credit_History']],Y_train)
# Checking the performance of our model on the testing data set
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome',
                             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))
Y_train.Target.value_counts()/Y_train.Target.count()
Y_test.Target.value_counts()/Y_test.Target.count()
# Importing MinMaxScaler and initializing it
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
#knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train_minmax,Y_train)
#accuracy_score(Y_test,knn.predict(X_test_minmax))
