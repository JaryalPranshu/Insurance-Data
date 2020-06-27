#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib


# In[5]:


df = pd.read_csv(r"C:\Users\pxj190011\Desktop\Projects\Medical\insurance.csv")


# In[6]:


df.head()


# In[7]:


print(df.shape)
print(df.describe())
print(df.dtypes)


# In[8]:


df.isnull().sum()


# In[9]:


df.corr()['charges'].sort_values()


# In[10]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)


# In[11]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=df)


# In[12]:


plt.figure(figsize=(10,5))
plt.title("Distribution of age")
ax = sns.distplot(df["age"], color = 'b')


# In[13]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", data=df[(df.age == 18)])
plt.title("The number of smokers and non-smokers (18 years old)")


# In[20]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges 18 years old smokers")
sns.boxplot(y="smoker", x="charges", data = df[(df.age == 18)] , orient="h", palette = 'pink')


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[14]:


X = df[['age', 'bmi', 'children']] 
y = df['charges']


# In[16]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)


# In[17]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=20);


# In[20]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)


# In[22]:


predictions = lm.predict(X_test)
predictions


# In[24]:


sns.distplot((y_test-predictions),bins=50)


# In[25]:


print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

