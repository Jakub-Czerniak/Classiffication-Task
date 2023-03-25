import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
y = y.map({'democrat':1, 'republican':-1})
y = y.values

for column in X:
    X[column] = X[column].map({'y':1, '?':0 , 'n':-1})

X.insert(0, "Ones", np.ones(len(X)))

r_state=3
training_data = X.sample(frac=0.6, random_state=r_state)
testing_data = X.drop(training_data.index)
validation_data = testing_data.sample(frac=0.5,random_state=r_state)
testing_data = testing_data.drop(validation_data.index)

weights = np.random.rand(len(X.columns))-0.5
learning_rate = 0.5

for index, row in training_data.iterrows():
    dot_product = np.dot(weights.transpose(),row)
    if(dot_product>0):
        prediction = 1
    else:
        prediction = -1
    if(prediction != y[index]):
        weights = weights + learning_rate * (y[index] - dot_product) * row 

democrat_tp=0
democrat_fp=0
democrat_tn=0
democrat_fn=0
success = 0
count = 0
predictionStr = ["","Democrat", "Republican"]
print(weights)
for index, row in validation_data.iterrows():
    dot_product = np.dot(weights.transpose(),row)
    if(dot_product>0):
        prediction = 1
    else:
        prediction = -1
    if(prediction == y[index]):
        success+=1
        if(y[index]==1):
            democrat_tp+=1
        elif(y[index]==-1):
            democrat_tn+=1
    elif(prediction==1):
        democrat_fp+=1
    else:
        democrat_fn+=1
    print('Row: ' + str(index) + ' Party: ' + predictionStr[y[index]] + ' Prediction: ' + predictionStr[prediction] )
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







    
