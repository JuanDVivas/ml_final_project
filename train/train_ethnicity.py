from random import random
import pandas as pd
import numpy as np 
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
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

X_train.loc[:,'labels'] = y_train
set_1 = X_train.sample(frac=0.5, replace=True, random_state=30)
set_2 = X_train.sample(frac=0.5, replace=True, random_state=42)
set_3 = X_train.sample(frac=0.5, replace=True, random_state=10)

pond=[]
##Logistic Regression

#Logistics Regression 
lr = LogisticRegression(max_iter=10000)
lr.fit(set_1.drop(columns=['labels']) , set_1.loc[: , 'labels'])
y_pred_lr = lr.predict(X_test)

print('Logistic Regression')
print('MAPE: ',mean_absolute_percentage_error(y_test,y_pred_lr))
print('R-Squared :', r2_score(y_test,y_pred_lr))

filename1 = 'lr_ethnicity.sav'
pickle.dump(lr, open(filename1, 'wb'))
pond.extend([r2_score(y_test,y_pred_lr)])

##Random Forest
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(set_2.drop(columns=['labels']) , set_2.loc[: , 'labels'])
y_pred_rf = rfc.predict(X_test)

print('Random Forest')
print('MAPE: ',mean_absolute_percentage_error(y_test,y_pred_rf))
print('R-Squared :', r2_score(y_test,y_pred_rf))

filename2 = 'rf_ethnicity.sav'
pickle.dump(rfc, open(filename2, 'wb'))
pond.extend([r2_score(y_test,y_pred_rf)])



##Neural Network:
nn = MLPRegressor(hidden_layer_sizes = (30,20,10), solver='adam', max_iter=10000)
nn.fit(set_3.drop(columns=['labels']), set_3.loc[:,'labels'])
y_pred_nn = nn.predict(X_test)

print('Neural Network')
print('MAPE: ',mean_absolute_percentage_error(y_test,y_pred_nn))
print('R-Squared :', r2_score(y_test,y_pred_nn))

filename3 = 'nn_ethnicity.sav'
pickle.dump(nn, open(filename3, 'wb'))
pond.extend([r2_score(y_test,y_pred_nn)])
