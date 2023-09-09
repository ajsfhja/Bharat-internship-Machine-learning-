#!/usr/bin/env python
# coding: utf-8

# # Bharat Internship-MachineLearning
# Wine Quality predicion 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


# In[16]:


wine = pd.read_csv(r"C:\Users\akhi\Downloads\winequalityN.csv")


# In[17]:


wine.head()


# In[18]:


pred_test = wine.iloc[3]


# In[19]:


pred_test['type'] = 1
pred_test.drop(['quality','total sulfur dioxide'],inplace=True)
#pred_test.drop('total_sulfur_dioxide',inplace=True)
pred_test


# In[20]:


wine.shape


# In[21]:


wine.isnull().sum()


# In[22]:


wine.describe()


# In[23]:


# One to remove na values is just by dropping them since they are very few
wine.dropna()
#another way is to impute let's say average value 
#wine.update(wine.fillna(wine.mean()))


# In[24]:


wine.info()


# In[26]:


wine['type'].value_counts()


# In[27]:


sns.countplot(x="type", data=wine)


# In[28]:


wine['type'].value_counts(normalize=True)


# In[29]:


#Checking distribution and outlier for each variable
plt.figure(2)
plt.subplot(121)
sns.distplot(wine['alcohol'])
plt.subplot(122)
wine['alcohol'].plot.box(figsize=(15,5))
#repeat this for all the variables and understand the distribution


# In[30]:


#bivariate analysis to check quality with all the other variables
plt.figure(figsize=(10,7))
sns.barplot(x='quality',y='alcohol',data=wine)


# In[31]:


#Plotting all variables for their distribution and relation
sns.pairplot(wine)


# In[32]:


#checking correlation
wine.corr()


# In[33]:


#buidling heatmap
plt.figure(figsize=(15,10))
sns.heatmap(wine.corr(), cmap='coolwarm')


# In[34]:


#Dropping highly correlated variables - in this case total sulfur dioxide
wine_new = wine.drop('total sulfur dioxide',axis=1)


# In[35]:


#Convert categorical value to dummies
wine_ml = pd.get_dummies(wine_new, drop_first=True)
wine_ml.head()


# In[36]:


wine_ml.dtypes


# In[37]:


wine_ml.dropna(inplace=True)
X = wine_ml.drop('quality',axis=1)


# In[38]:


X.isnull().sum()


# In[39]:


Y = wine_ml['quality'].apply(lambda y: 1 if y > 7 else 0)
Y


# In[40]:


scaler = StandardScaler()
scaler.fit(X)
x_standard = scaler.transform(X)


# In[41]:


scaler = StandardScaler()
pred_test = np.asarray(pred_test).reshape(1,-1)
scaler.fit(pred_test)
pred_test_std = scaler.transform(pred_test)


# In[42]:


X = x_standard


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,random_state=123)


# In[45]:


logreg = LogisticRegression()


# In[46]:


logreg.fit(X_train, Y_train)


# In[47]:


y_pred = logreg.predict(X_test)


# In[48]:


pred_test_output = logreg.predict(pred_test_std)
pred_test_output


# In[49]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(Y_test, y_pred)


# In[50]:


print(classification_report(Y_test, y_pred))


# In[51]:


confusion_matrix(Y_test, y_pred)


# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


rfc = RandomForestClassifier(n_estimators=200)


# In[54]:


rfc.fit(X_train, Y_train)


# In[55]:


rfc_pred = rfc.predict(X_test)


# In[56]:


accuracy_score(Y_test, rfc_pred)


# In[57]:


confusion_matrix(Y_test, rfc_pred)


# In[58]:


classification_report(Y_test, rfc_pred)


# In[59]:


print(classification_report(Y_test, rfc_pred))


# In[60]:


rfc.feature_importances_


# In[61]:


pd.Series(rfc.feature_importances_,index=wine_ml.drop('quality',axis=1).columns).plot(kind='barh')


# In[ ]:




