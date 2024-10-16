# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: start the program

Step 2: Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

Step 3: Split the data into training and test sets using train_test_split.

Step 4: Create and fit a logistic regression model to the training data.

Step 5: Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

Step 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

Step 7:End the program.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DURGA V
RegisterNumber: 212223230052 
*/

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("Placement_Data_Full_Class (1).csv")
df.head()
```

## Output:
![image](https://github.com/user-attachments/assets/f456e70a-96e6-4a58-83b8-9b99a70ab168)

## Program:
```
df.tail()
```
## output:
![image](https://github.com/user-attachments/assets/493f0a81-cd25-4b07-8272-736cc6c26de5)

## Program:
```
df.drop('sl_no',axis=1)
```
## Output:
![image](https://github.com/user-attachments/assets/d548c35f-5dd5-4853-be35-f6279f36fbbb)

## Program:
```
df.drop('sl_no',axis=1,inplace=True)
```
```
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df.dtypes
```
## Output:
![image](https://github.com/user-attachments/assets/d02e525b-1303-482a-a457-b7331cc49da6)

## Program:
```
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes

df.info()

df.head()

```

## Output:
![image](https://github.com/user-attachments/assets/18209dc6-e069-4d82-9877-ce4354923921)

## Program:
```
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

Y
```
## Output:
![image](https://github.com/user-attachments/assets/837329d2-ea7b-4a1c-8894-ec62e9637dab)

```
X
```
## Output:
![image](https://github.com/user-attachments/assets/5f50343e-e6b2-4e3b-91b0-d82057c8ae38)

```
X.shape
```
## Output:
![image](https://github.com/user-attachments/assets/f4f5a77d-b423-4099-b8f5-f60f31042220)

```
Y.shape
```
## Output:
![image](https://github.com/user-attachments/assets/ef648be5-caf5-4cd9-85ad-fac3cf14d0c8)

```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

X_train.shape
```
## Output:
![image](https://github.com/user-attachments/assets/d2e60f0a-e1a6-409a-aff5-52cd63a83d5b)

```
Y_train.shape
```
## Output:
![image](https://github.com/user-attachments/assets/53383797-8ddc-4ff6-97a0-de93d854b3b7)

```
Y_test.shape
```
## Output:
![image](https://github.com/user-attachments/assets/70789ba7-c40c-4495-b20b-16c60e17d781)

```
clf = LogisticRegression()
clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cf = confusion_matrixaccuracy_score
print(cf)
```
## output:
![image](https://github.com/user-attachments/assets/04a5c989-0135-43c6-a7be-af1dd9e9cafd)

```
accuracy=accuracy_score(Y_test,Y_pred)
print(accuracy)
```
## output:
![image](https://github.com/user-attachments/assets/68e0f4ea-60d2-45f6-b69b-5e08395ef19c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
