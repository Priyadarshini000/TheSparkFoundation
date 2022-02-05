#!/usr/bin/env python
# coding: utf-8

# # Prediction Using Supervised Machine Learning
# 
# 

# # Submitted By:- Priyadarshini PAL
# 

# # Objective:-Predict the percentage of student based on number of study hours using simple Linear regression

# # Step 1: To install all the required libraries

# In[1]:


pip install sklearn


# In[2]:


pip install matplotlib


# In[4]:


pip install sqrt


# In[5]:


pip install cov


# In[69]:


pip install numpy


# # Step2: Import all the required Libraries

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import numpy as np
from math import sqrt


# # Step3: Read the data

# In[7]:


df=pd.read_csv("C:/Users/Dell/Desktop/sparkstudent.csv")
 #displaying the top 20 values in the dataset
df.head(20)


# In[8]:


#For any 5 sample from data
df.sample(5)


# # Step4: Get the data knowledge 

# In[9]:


# For the Number of rows and columns in the dataset
df.shape


# In[10]:


# Data overall description
df.describe()


# In[11]:


#Checking whether there are any missing values in our data
df.isna().sum()


# # Step5: Analyzing the data fro better predition

# In[12]:


#for analyzing the scores with increasing noumber of study hours
df.groupby(['Hours']).mean()


# In[13]:


#for analyzing the hours with increasing noumber of scores
df.groupby(['Scores']).mean()


# # Step 6: Visualizing the given Data

# In[14]:


#Histogram of Hours
plt.style.use('ggplot')

df.Hours.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Study Hours', size=24)
plt.xlabel('Study Hours ', size=18)
plt.ylabel('Frequency ', size=18)


# In[15]:


#Histogram of Scores
plt.style.use('ggplot')

df.Scores.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Scores', size=24)
plt.xlabel('Scores ', size=18)
plt.ylabel('Frequency ', size=18)


# In[16]:


df.plot(x='Scores',y='Hours', style='*', color="g")
plt.xlabel("Scores Obtained")
plt.ylabel("Study Hours")
plt.title("Scores Obtained Vs Study Hours")
plt.show()


# # Step7:Split the Dataset into Training Set and Testing Set for Prediction

# In[17]:


hour=df.iloc[:, :1].values # for array of only Hours
score=df.iloc[:, 1:].values # for array of only Scores


# In[18]:


hour


# In[19]:


score


# In[51]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(hour,score,train_size=0.70,test_size=0.30, random_state=0)


# In[52]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_train)


# In[ ]:





# # Step8: Plotting Simple Linaer Regression Line with the given data

# In[53]:


line= model.coef_*hour + model.intercept_
plt.scatter(hour,score)
plt.plot(hour,line,color='b')
plt.xlabel("Study Hours")
plt.ylabel("Scores obtained")
plt.title("Study Hours vs Scores Obtained")
plt.show()


# # Step9: Prediction of Scores based on Study hours

# In[56]:


y_predict= model.predict(x_test)
y_predict


# In[57]:


#Actual values 
print("Actual values: ")
y_test


# In[58]:


# predicted values 
print("Prediction values: ")
y_predict


# In[59]:


#Comparing actual and predicted values
df1= pd.DataFrame({'Actual':[y_test],'Predicted':[y_predict]})
df1


# In[60]:


# Prediction of scores for 9.25 hours
print("Score of a student for 9.25 study hours is ",model.predict([[9.25]]))


# # Step10: Accuracy of the Model

# In[61]:


absolute_error=metrics.mean_absolute_error(y_test,y_predict)
MeanSq_error=metrics.mean_squared_error(y_test,y_predict)
print("Mean Squared Error for the Model: ",MeanSq_error )
print("Mean Absolute Error for the Model: ",absolute_error )


# In[62]:


rms = sqrt(mean_squared_error(y_test,y_predict))
print(rms)


# In[76]:


np.cov(df)


# In[67]:



regr = linear_model.LinearRegression()
regr.fit(hour, score)

print(regr.coef_)


# In[ ]:




