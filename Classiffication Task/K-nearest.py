import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

data = pd.read_csv('Data/house-votes-84.data', sep=",", header=None, index_col=False)
data.columns=["democrat, republican",
   "handicapped-infants(y,n)",
   "water-project-cost-sharing(y,n)",
   "adoption-of-the-budget-resolution(y,n)",
   "physician-fee-freeze(y,n)",
   "el-salvador-aid(y,n)",
   "religious-groups-in-schools(y,n)",
   "anti-satellite-test-ban(y,n)",
   "aid-to-nicaraguan-contras(y,n)",
  "mx-missile(y,n)",
  "immigration(y,n)",
  "synfuels-corporation-cutback(y,n)",
  "education-spending(y,n)",
  "superfund-right-to-sue(y,n)",
  "crime(y,n)",
  "duty-free-exports(y,n)",
  "export-administration-act-south-africa(y,n)"]

data["democrat, republican"].hist(bins=3)
#plt.show()

X = data.drop("democrat, republican", axis=1)
y = data["democrat, republican"]

for column in X:
    X[column] = X[column].map({'y':1, '?':0 , 'n':-1})

r_state=3
training_data = X.sample(frac=0.6, random_state=r_state)
testing_data = X.drop(training_data.index)
validation_data = testing_data.sample(frac=0.5,random_state=r_state)
testing_data = testing_data.drop(validation_data.index)

y_training = y.sample(frac=0.6,random_state=r_state)
y_testing = y.drop(y_training.index)
y_validation = y_testing.sample(frac=0.5,random_state=r_state)
y_testing = y_testing.drop(y_validation.index)

y_training = y_training.values
y_testing = y_testing.values
y = y.values

#print(str(len(testing_data.index)))
#print(str(len(training_data.index)))
#print(str(len(validation_data.index)))

success=0
count=0
democrat_tp=0
democrat_fp=0
democrat_tn=0
democrat_fn=0
for index, row in validation_data.iterrows():
    distances = np.linalg.norm(training_data - row, axis=1)
    k = 5
    nearest_neighbours_ids = distances.argsort()[:k]
    nearest_neighbours_party = y_training[nearest_neighbours_ids]
    prediction = scipy.stats.mode(nearest_neighbours_party, keepdims=False)
    if(prediction.mode == y[index]):
        success+=1
        if(y[index]=='democrat'):
            democrat_tp+=1
        elif(y[index]=='republican'):
            democrat_tn+=1
    elif(prediction.mode=='democrat'):
        democrat_fp+=1
    else:
        democrat_fn+=1
    print('Row: ' + str(index) + ' Party: ' + y[index] + ' Prediction: ' + prediction.mode  )
    count+=1

success_rate=success/count
print('Success count: ' + str(success))
print('Overall count: ' + str(count))
print('Success rate: ' + str(int(success_rate*100)) + '%')

print('Democrat')
print('True positives: ' + str(democrat_tp))
print('False positives: ' + str(democrat_fp))
print('True negatives: ' + str(democrat_tn))
print('False negatives: ' + str(democrat_fn))

republican_tp=democrat_tn
republican_fp=democrat_fn
republican_tn=democrat_tp
republican_fn=democrat_fp

print('Republican')
print('True positives: ' + str(republican_tp))
print('False positives: ' + str(republican_fp))
print('True negatives: ' + str(republican_tn))
print('False negatives: ' + str(republican_fn))