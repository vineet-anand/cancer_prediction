#!/usr/bin/env python
# coding: utf-8

# # Cancer Feature Prediction

# ## Importing the libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the dataset


df = pd.read_csv('cancer.csv')
df.head()


# ## Data Cleaning



df.isna().sum()




# M = Malignant (Cancer detected), B = Benign (No Cancer)
df['diagnosis'].value_counts()




sns.countplot(df['diagnosis'])



df.dtypes


# ## Enocoding categorical data 



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()




df.iloc[:,1] = le.fit_transform(df.iloc[:,1].values)


# ## Analyzing Data (only for features having a mean values)



sns.pairplot(df.iloc[:,1:6], hue = 'diagnosis')




df.iloc[:,1:12].corr()




plt.figure(figsize=(8,8))
sns.heatmap(df.iloc[:,1:12].corr(), annot = True, fmt = '.0%')


# ##  Dataset Splitting (Training set & Test set)



X = df.iloc[:,2:].values
y = df.iloc[:,1].values




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state= 0)


# ## Feature Scaling



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()




X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)




print(np.unique(X_train))
print(X_train.dtype)




print(np.unique(X_test))
print(X_test.dtype)




print(np.unique(y_train))
print(y_train.dtype)




print(np.unique(y_test))
print(y_test.dtype)




y_train = y_train.astype(int)
y_test = y_test.astype(int)




print(np.unique(y_train))
print(y_train.dtype)




print(np.unique(y_test))
print(y_test.dtype)


# ## Model Training



def models(X_train, y_train):
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)
    
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)
    
    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, y_train)
    
    # Model Accuracy
    print('Logistic Regression:', log.score(X_train, y_train))
    print('Decision Tree Regression:', tree.score(X_train, y_train))
    print('Random Forest Regression:', forest.score(X_train, y_train))

    return log, tree, forest




model = models(X_train, y_train)


# ## Model Testing on Confusion Matrix & Accuracy Score



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
for i in range(len(model)):
    
    cm = confusion_matrix(y_test, model[i].predict(X_test))
    report = classification_report(y_test, model[i].predict(X_test))
    #display = ConfusionMatrixDisplay(cm).plot()
    score = accuracy_score(y_test, model[i].predict(X_test))

    print('Model: ',model[i])
    print(cm)
    print('Report:')
    print(report)
    print('Accuracy:', score*100)
    print()
    


# ## Predicting the Test set results



for i in range(len(model)):
    y_pred = model[i].predict(X_test)
    print(model[i])
    print(y_test)
    print(y_pred)


# ## Computing the accuracy with k-Fold Cross Validation



from sklearn.model_selection import cross_val_score

for i in range(len(model)):
    accuracies = cross_val_score(estimator = model[i], X = X_train, y = y_train, cv = 10)
    print(model[i])
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))





