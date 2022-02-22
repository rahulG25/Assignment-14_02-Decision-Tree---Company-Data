#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#for encoding
from sklearn.preprocessing import LabelEncoder#for train test splitting
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree 
from sklearn.tree import plot_tree


# In[3]:


#reading the data
df = pd.read_csv('Company_Data.csv')
df.head()


# In[5]:


#getting information of dataset
df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().any()


# In[8]:


# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df, hue = 'ShelveLoc')


# In[9]:


#Creating dummy vairables by dropping first dummy variable
df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)


# In[10]:


print(df.head())


# In[11]:


df.info()


# In[13]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[14]:


df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})


# In[15]:


print(df.head())


# In[16]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']


# In[17]:


x


# In[18]:


y


# In[19]:


df['ShelveLoc'].unique()


# In[20]:


df.ShelveLoc.value_counts()


# In[21]:


colnames = list(df.columns)
colnames


# In[22]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# # Building Decision Tree Classifier using Entropy Criteria

# In[23]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[24]:


from sklearn import tree


# In[25]:


#PLot the decision tree
tree.plot_tree(model);


# In[26]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[27]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[28]:


preds


# In[29]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[30]:


# Accuracy 
np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[31]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[32]:


model_gini.fit(x_train, y_train)


# In[33]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# Decision Tree Regression

# In[34]:


from sklearn.tree import DecisionTreeRegressor


# In[35]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[37]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[38]:


#Find the accuracy
model.score(X_test,y_test)

