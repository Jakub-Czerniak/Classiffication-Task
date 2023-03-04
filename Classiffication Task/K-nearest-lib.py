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

training_data = X.sample(frac=0.8, random_state=3)
testing_data = X.drop(training_data.index)
y_training = y.sample(frac=0.8,random_state=3)
y_testing = y.drop(y_training.index)
y_training = y_training.values
y_testing = y_testing.values
y = y.values

success=0
count=0
for index, row in testing_data.iterrows():
    distances = np.linalg.norm(training_data - row, axis=1)
    k = 5
    nearest_neighbours_ids = distances.argsort()[:k]
    #print(index)
    #print(nearest_neighbours_ids)
    #print(distances)
    nearest_neighbours_party = y_training[nearest_neighbours_ids]
    #print(nearest_neighbours_party)
    prediction = scipy.stats.mode(nearest_neighbours_party, keepdims=False)
    if(prediction.mode == y[index]):
        success+=1
    print('Row: ' + str(index) + ' Party: ' + y[index] + ' Prediction: ' + prediction.mode  )
    count+=1

success_rate=success/count
print('Success count: ' + str(success))
print('Overall count: ' + str(count))
print('Success rate: ' + str(int(success_rate*100)) + '%')