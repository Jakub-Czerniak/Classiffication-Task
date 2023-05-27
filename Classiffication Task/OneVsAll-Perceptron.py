import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('IrisData/iris.data', sep=",", header=None, index_col=False)
data.columns=["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class" ]

data["class"].hist(bins=5)
#plt.show()

X=data.drop("class", axis=1)
X.insert(0, "Ones", np.ones(len(X)))

r_state = 8
training_data= X.sample(frac=0.6, random_state=r_state)
validation_data = X.drop(training_data.index)
testing_data = validation_data.sample(frac=0.5,random_state= r_state)
validation_data = validation_data.drop(testing_data.index)

y=data["class"]
y_setosa=y.map({'Iris-setosa':1, 'Iris-versicolor':-1, 'Iris-virginica':-1 })
y_versicolor=y.map({'Iris-setosa':-1, 'Iris-versicolor':1, 'Iris-virginica':-1 })
y_virginica=y.map({'Iris-setosa':-1, 'Iris-versicolor':-1, 'Iris-virginica':1 })
y=y.map({'Iris-setosa':-1, 'Iris-versicolor':0, 'Iris-virginica':1 })
y_setosa = y_setosa.values
y_versicolor = y_versicolor.values
y_virginica = y_virginica.values
y=y.values


weights_setosa = weights_virginica = weights_versicolor = np.random.rand(len(X.columns))-0.5
learning_rate = 0.01


for index, row in training_data.iterrows():
    dot_product_setosa = np.dot(weights_setosa.transpose(),row)
    dot_product_virginica = np.dot(weights_virginica.transpose(),row)
    dot_product_versicolor = np.dot(weights_versicolor.transpose(),row)

    if(dot_product_setosa>0):
        prediction_setosa = 1
    else:
        prediction_setosa = -1

    if(dot_product_virginica>0):
        prediction_virginica= 1
    else:
        prediction_virginica= -1

    if(dot_product_versicolor>0):
        prediction_versicolor= 1
    else:
        prediction_versicolor= -1

    if(prediction_setosa != y_setosa[index]):
        weights_setosa = weights_setosa + learning_rate * (y_setosa[index] - prediction_setosa) * row

    if(prediction_virginica != y_virginica[index]):
        weights_virginica = weights_virginica + learning_rate * (y_virginica[index] - prediction_virginica) * row

    if(prediction_versicolor != y_versicolor[index]):
        weights_versicolor = weights_versicolor + learning_rate * (y_versicolor[index] - prediction_versicolor) * row

    
setosa_tp=0
setosa_tn=0
setosa_fp=0
setosa_fn=0
versicolor_tp = 0
versicolor_tn = 0
versicolor_fn = 0
versicolor_fp = 0
virginica_tp = 0
virginica_tn = 0
virginica_fp = 0
virginica_fn = 0
success = 0
count = 0
predictionStr = ["Iris Versicolor", "Iris Virginica" ,"Iris Setosa"]
for index, row in validation_data.iterrows():
    dot_product_setosa = np.dot(weights_setosa.transpose(),row)
    dot_product_virginica = np.dot(weights_virginica.transpose(),row)
    dot_product_versicolor = np.dot(weights_versicolor.transpose(),row)
    #print(dot_product_setosa)
    #print(dot_product_virginica)
    #print(dot_product_versicolor)
    if(dot_product_setosa > dot_product_virginica and dot_product_setosa > dot_product_versicolor):
        prediction = -1
    elif(dot_product_setosa < dot_product_virginica and dot_product_versicolor < dot_product_virginica):
        prediction = 1
    else:
        prediction = 0

    if(prediction == y[index]):
        success+=1
        if(prediction==-1):
            setosa_tp+=1
            versicolor_tn+=1
            virginica_tn+=1
        elif(prediction==0):
            versicolor_tp+=1
            setosa_tn+=1
            virginica_tn+=1
        else:
            virginica_tp += 1
            setosa_tn+=1
            versicolor_tn+=1
    elif(prediction==1):
        virginica_fp+=1
        if(y[index]==0):
            versicolor_fn+=1
            setosa_tn+=1
        else:
            setosa_fn+=1
            versicolor_tn+=1
    elif(prediction==0):
        versicolor_fp+=1
        if(y[index]==1):
            virginica_fn+=1
            setosa_tn+=1
        else:
            setosa_fn+=1
            virginica_tn+=1
    else:
        setosa_fp+=1
        if(y[index]==0):
            versicolor_fn+=1
            virginica_tn+=1
        else:
            virginica_fn+=1
            versicolor_tn+=1
    print('Row: ' + str(index) + ' Party: ' + predictionStr[y[index]] + ' Prediction: ' + predictionStr[prediction] )
    count+=1

success_rate=success/count
print('Success count: ' + str(success))
print('Overall count: ' + str(count))
print('Success rate: ' + str(int(success_rate*100)) + '%')

print('Iris virginica')
print('True positives: ' + str(virginica_tp))
print('False positives: ' + str(virginica_fp))
print('True negatives: ' + str(virginica_tn))
print('False negatives: ' + str(virginica_fn))

print('Iris setosa')
print('True positives: ' + str(setosa_tp))
print('False positives: ' + str(setosa_fp))
print('True negatives: ' + str(setosa_tn))
print('False negatives: ' + str(setosa_fn))

print('Iris versicolor')
print('True positives: ' + str(versicolor_tp))
print('False positives: ' + str(versicolor_fp))
print('True negatives: ' + str(versicolor_tn))
print('False negatives: ' + str(versicolor_fn))
