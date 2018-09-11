
# coding: utf-8

# In[34]:


import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import itertools
from sklearn.metrics import confusion_matrix


from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier)


# In[35]:


#shapes_ is a 2D array containing the image files
twenty = []
thirty = []
fifty = []
sixty = []
seventy = []
eighty = []
#hundred = []
#onetwenty = []
#nocross = []
#nocross1 = []
#nocross2 = []
streetsigns  =["twenty","thirty","fifty",'sixty','seventy',"eighty"]
streetsigns_ = [twenty,thirty, fifty, sixty,seventy, eighty]


# In[36]:


images = []
noisy =[]
for i,sign in enumerate(streetsigns):
    
    path = "E:\\streetsigns/"+sign
    files = [f for f in os.listdir(path) if isfile(join(path, f))]

    for file in files:
        
        fpath =path+"/"+file
        img=cv.imread(fpath,0)
        # here i do a PCA to convert the Image file into a vector
        images.append(img)
        pca = PCA(n_components=1)
        img = pca.fit_transform(img)
        newimg = []
        for cell in img:
            newimg.append(cell[0])        
        streetsigns_[i].append(newimg)


# In[37]:


#creating a Dataframe

dftw = pd.DataFrame(streetsigns_[0])   
dftw["sign"]= [1 for c in range(len(dftw))]

dfth = pd.DataFrame(streetsigns_[1])   
dfth["sign"]= [2 for c in range(len(dfth))]

dffi = pd.DataFrame(streetsigns_[2])   
dffi["sign"]= [3 for c in range(len(dffi))]

dfsi = pd.DataFrame(streetsigns_[3])   
dfsi["sign"]= [4 for c in range(len(dfsi))]

dfse = pd.DataFrame(streetsigns_[4])   
dfse["sign"]= [5 for c in range(len(dfse))]

dfei = pd.DataFrame(streetsigns_[5])   
dfei["sign"]= [6 for c in range(len(dfei))]

df = pd.concat([dftw,dfth, dffi, dfsi,dfse,dfei],ignore_index=True)


# In[38]:


#prepare for classification
X = df.drop(["sign"], axis=1)
y = df["sign"]
#look at PCA in 2D


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=10,stratify=y)
# using Random Forest CLassification
classifier = RandomForestClassifier()


clf = classifier
clf = clf.fit(X_train, y_train)
print ("Mean accuracy of the model is", clf.score(X_test, y_test))


# In[40]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt ="d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Sign')
    plt.xlabel('Predicted Sign')


# In[41]:


#little help function to measure accuracy
def predict_acc(X_test,y_test,clf):
    predictions=[]                      
    for i in range(len(X_test)): 
        predictions.append(float(clf.predict([X_test.iloc[i]])[0]))   
    acc = accuracy_score(y_test, predictions)
    return [acc,predictions]


# In[42]:


class_names = streetsigns
y_pred = predict_acc(X_test,y_test,clf)[1]
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()


# In[43]:


n=100
labels = ["RandomForest","AdaBoost"]
classifiers =  [RandomForestClassifier,
                              AdaBoostClassifier]
accs_classifier = []
for j,c in enumerate(classifiers):
    
    accs = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)
        clf = RandomForestClassifier()
        clf = clf.fit(X_train, y_train)
        acc = predict_acc(X_test,y_test,clf)[0]
        accs.append(acc)
    accs_classifier.append(accs)
    
    
    print("Mean accuracy of "+labels[j]+" "+str(np.mean(accs)))

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
plt.title("Distribution of Accuracy of 2 Classifiers")
ax1.boxplot(accs_classifier,labels=labels)

