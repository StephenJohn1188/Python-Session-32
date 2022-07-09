#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')


# In[14]:


cancerset=load_breast_cancer()
cancerset


# In[15]:


cancerset.keys()


# In[17]:


cancerset.data


# In[18]:


cancerset.target


# In[21]:


cancerset.feature_names


# In[23]:


cancerset.DESCR


# In[24]:


df=pd.DataFrame(data=cancerset.data)


# In[25]:


df


# In[28]:


df=pd.DataFrame(data=cancerset.data, columns=cancerset.feature_names)


# In[29]:


df


# In[31]:


df['target']=pd.DataFrame(cancerset.target)
df


# In[32]:


df.info()


# In[33]:


df.describe()


# In[34]:


df.head()


# In[35]:


df.tail(10)


# In[36]:


df.sample()


# In[37]:


df.corr()


# In[39]:


sns.heatmap(df.corr(),annot=True)


# In[44]:


sns.countplot(df['target'])


# In[45]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)


# In[46]:


x=df.drop(['target'],axis=1)


# In[47]:


x


# In[48]:


y=df['target']


# In[49]:


y


# In[51]:


x.shape


# In[52]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=20,random_state=45)


# In[53]:


train_x


# In[54]:


test_x


# In[55]:


train_y


# In[56]:


test_y


# In[57]:


lg=LogisticRegression()


# In[58]:


lg.fit(train_x,train_y)


# In[59]:


pred=lg.predict(test_x)


# In[61]:


print(pred)


# In[62]:


print('Accuracy Score=',accuracy_score(test_y,pred))


# In[64]:


print(confusion_matrix(test_y,pred))


# In[66]:


print(classification_report(test_y,pred))

