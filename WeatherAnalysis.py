
# coding: utf-8

# In[ ]:


##Analysis of weather
#Importing Required libraries and importing data.


# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r'C:\Users\User\Desktop\daily_weather.csv', header = 0)

data.columns


# In[2]:


data


# In[ ]:


#Cleaning of the data 


# In[3]:


data[data.isnull().any(axis=1)]

del data['number']


# In[4]:


before_rows = data.shape[0]
print(before_rows)

data = data.dropna()



after_rows = data.shape[0]
print(after_rows)


before_rows - after_rows


# In[ ]:


#Setting a binary value for relative humidity. ie. Either true or false.


# In[5]:


clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
print(clean_data['high_humidity_label'])


# In[ ]:


#Creating a new table for relative humidity at 3PM.


# In[7]:


y=clean_data[['high_humidity_label']].copy()


clean_data['relative_humidity_3pm'].head()


# In[8]:


y.head()


# In[9]:


clean_data


# In[ ]:


#Creating new table with features at 9 Am.


# In[10]:


morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']


# In[11]:


X = clean_data[morning_features].copy()
X
X.columns


# In[12]:


y.columns


# In[ ]:


#Creating test and train data frames


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)


# In[14]:


type(humidity_classifier)


# In[15]:


predictions = humidity_classifier.predict(X_test)
predictions[:10]


# In[16]:


y_test['high_humidity_label'][:10]


# In[ ]:


#Calculating accuracy of Prediction. Score can be varied by varying test size and random state variables.


# In[17]:


accuracy_score(y_true = y_test, y_pred = predictions)

