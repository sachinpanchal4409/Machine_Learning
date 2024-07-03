# Import required libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle
# Download Dataset into dataset variable
dataset=pd.read_csv("D:\project\Diabetese/diabetes_prediction_dataset.csv")
print("Shape of Dataset: ", dataset.shape)
# To categories the columns, use one hot encoding techniques
one_hot_encode_dataset= pd.get_dummies(dataset,columns=["gender","smoking_history"])
scaler = StandardScaler().fit(one_hot_encode_dataset)
X, y = make_classification(random_state=42)

X=one_hot_encode_dataset.drop("diabetes",axis=1)
y=one_hot_encode_dataset["diabetes"]

# Divide whole dataset into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25,shuffle=True)

X_train.head()
model = make_pipeline(StandardScaler(), LogisticRegression())
# Use classifier for classification 
#model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train,y_train)

#pickle.dump(model,open(model.pkl, 'wb'))

#model= model.load(open(model.pkl,'rb'))
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score
acc=accuracy_score(y_test,y_pred)
print("Accuracy=",acc)
f1_score=f1_score(y_test,y_pred)
print("f1_score",f1_score)
