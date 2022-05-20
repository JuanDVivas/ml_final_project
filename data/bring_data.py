import pandas as pd
import numpy as np 
import sys
import matplotlib.pyplot as plt
#sys.path.insert( 0, '../../lib/python3' )

#Load Data:
data = pd.read_csv(open( sys.argv[ 1 ], 'rb' ))
print(data.describe())

for i in data.columns:
    if i == 'ethnicity':
        print(data.loc[:,i].value_counts())
    else:
        pass
        
print(data.dtypes)

#Pixels to list:
data.loc[:,'pixels'] = data.loc[:,'pixels'].apply(lambda x: np.array(x.split(), dtype="float64"))

#Data Writting:
fraction = 0.2
data = data.sample(frac=1, random_state=30)

limit =  int(len(data)*fraction)

train_ = data.iloc[limit:]
test_ = data.iloc[:limit]

print('Index Limit Train-Test:', limit)

## Labels:
for i in ['age','ethnicity','gender']:
    #Train:
    y_temp_tr = train_.loc[:,i].to_numpy(dtype="float64")
    np.save('y_%s_train'%i, y_temp_tr)
    print('Data :', 'y_%s_train'%i, 'Successfully Written ', y_temp_tr.shape)

    #Test:
    y_temp_ts = test_.loc[:,i].to_numpy(dtype="float64")
    np.save('y_%s_test'%i, y_temp_ts)
    print('Data :', 'y_%s_test'%i, 'Successfully Written ', y_temp_ts.shape)


## Obs:
### Train:
img_size = train_.loc[:,'pixels'].iloc[0].reshape(1,len(train_.loc[:,'pixels'].iloc[0])).shape[1]
x_train = []

for i in range(len(train_)):
    x_train.append(list(train_.loc[:,'pixels'].iloc[i]))

x_train = np.reshape(x_train, newshape=(len(train_),img_size))
np.save('x_train', x_train)
print('Data :', 'X_train' , 'Successfully Written ', x_train.shape)

### Test:
x_test = []

for i in range(len(test_)):
    x_test.append(list(test_.loc[:,'pixels'].iloc[i]))

x_test = np.reshape(x_test, newshape=(len(test_),img_size))
np.save('x_test', x_test)
print('Data :', 'X_test' , 'Successfully Written ', x_test.shape)

#Show Random Image:
index_ini = int(sys.argv[ 2 ])
index_fin = int(sys.argv[ 3 ])

plt.figure(figsize=(16,16))
for i in range(index_ini,index_fin):
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data.loc[:,'pixels'].iloc[i].reshape(48,48), cmap='gray')
    plt.xlabel(
        "Age:"+str(data['age'].iloc[i])+
        "  Ethnicity:"+str(data['ethnicity'].iloc[i])+
        "  Gender:"+ str(data['gender'].iloc[i])
    )
plt.show()
