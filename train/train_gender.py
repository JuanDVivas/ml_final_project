from random import random
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.metrics import classification_report,balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

#Load Data:
X_train = np.load(open( sys.argv[ 1 ], 'rb' ))
X_test = np.load(open( sys.argv[ 2 ], 'rb' ))
y_train = np.load(open( sys.argv[ 3 ], 'rb' ))
y_test = np.load(open( sys.argv[ 4 ], 'rb' ))

################################################################################
#Process Data:
##Standardize:
def standardize(array, mean, std):
  return (array - mean) / std

### Store Tokenizer:
meanV = np.mean(X_train, axis=0)
S = np.std(X_train, axis=0) + 1e-12

np.save('tokenizer_mean', meanV)
np.save('tokenizer_std', S)

### Apply Standardization:
X_train = standardize(X_train, meanV, S)
X_test = standardize(X_test, meanV, S)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

y_train.rename(columns = {0:'labels'}, inplace = True)
y_test.rename(columns = {0:'labels'}, inplace = True)

X_train.loc[:,'labels'] = y_train
set_1 = X_train.sample(frac=0.5, replace=True, random_state=30)
set_2 = X_train.sample(frac=0.5, replace=True, random_state=42)
set_3 = X_train.sample(frac=0.5, replace=True, random_state=10)

################################################################################
#Modeling:
pond = []

##Logistic Regression:

#train_gender.py x_train.npy  x_test.npy y_gender_train.npy y_gender_test.npy

num = 60000
log_reg = LogisticRegression(solver = 'saga', l1_ratio = 0 ,penalty= 'elasticnet', max_iter = num)
log_reg.fit(set_1.drop(columns=['labels']), set_1.loc[:,'labels'])
y_pred_log_reg = log_reg.predict(X_test)

print('Regresión Logistica')
print('Accuracy:', accuracy_score(y_test,y_pred_log_reg))

filename1 = 'log_reg_gender.sav'
pickle.dump(log_reg, open(filename1, 'wb'))
pond.extend([r2_score(y_test,y_pred_log_reg)])

#SVM

svm = SVC()
svm.fit(set_2.drop(columns=['labels']), set_2.loc[:,'labels'])
y_pred_svm = svm.predict(X_test)

print('SVM')
print('Accuracy:', accuracy_score(y_test,y_pred_svm))

filename2 = 'svm_gender.sav'
pickle.dump(svm, open(filename2, 'wb'))
pond.extend([r2_score(y_test,y_pred_svm)])


##Neural Network:
nn = MLPClassifier(hidden_layer_sizes = (30,20,10), solver='adam', max_iter=10000)
nn.fit(set_3.drop(columns=['labels']), set_3.loc[:,'labels'])
y_pred_nn = nn.predict(X_test)

print('Neural Network')
print('Accuracy:', accuracy_score(y_test,y_pred_log_reg))
print('Balanced_accuracy :',  balanced_accuracy_score(y_test,y_pred_log_reg))
filename3 = 'nn_gender.sav'
pickle.dump(nn, open(filename3, 'wb'))
pond.extend([accuracy_score(y_test,y_pred_nn)])


