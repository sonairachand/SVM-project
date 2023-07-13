#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('C:/Users/unni/Desktop/Social_Network_Ads.csv')
dataset.head()


# In[3]:


x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[6]:


from sklearn.svm import SVC
classifier_rbf = SVC(kernel = 'rbf', random_state = 0)
classifier_rbf.fit(x_train, y_train)


# In[7]:


y_pred_rbf = classifier_rbf.predict(x_test)


# In[8]:


from sklearn.metrics import confusion_matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
print(cm_rbf)


# In[9]:


from sklearn.metrics import classification_report
class_report_rbf= classification_report (y_test,y_pred_rbf)
print(class_report_rbf)


# In[ ]:




