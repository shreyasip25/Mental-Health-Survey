#!/usr/bin/env python
# coding: utf-8

# # Classification of observed mental health consequences

# In[285]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# # Reading csv file
# 

# In[287]:


df = pd.read_csv('C:/Users/HP/Desktop/Predective Analytics Assignment/survey.csv')


# In[288]:


df


# In[269]:


df.info()


# In[270]:


df.head()


# In[271]:


df.describe()


# In[272]:


df.shape
#gives rows and column count


# In[273]:


#Checking null values


# In[274]:


df.isnull().sum()


# In[275]:


#Cleaning null values


# In[276]:


df['self_employed'].isnull()


# In[277]:


#converting into numeric values


# In[278]:


df['self_employed']=df['self_employed'].replace("Yes",int(1))
df['self_employed']=df['self_employed'].replace("No",int(0))


# In[279]:


df


# In[280]:


df['self_employed'].mode().astype(int)


# In[264]:


df


# In[265]:


df['self_employed']=df['self_employed'].fillna(1)


# In[281]:


df


# In[282]:


df["Gender"] =df['Gender'] .replace("Male",0)
df["Gender"] =df['Gender'] .replace("Female",1)


# In[283]:


df


# In[284]:


df["Gender"] =df['Gender'] .replace("Male ",0)


# In[197]:


df


# In[198]:


df["Gender"][1234]


# In[199]:


df["Gender"] =df["Gender"] .replace(" Male",0)


# In[200]:


df


# In[201]:


df["Gender"]= df["Gender"] =df['Gender'] .replace("Guy  ",0)
df["Gender"]= df["Gender"] =df['Gender'] .replace("Female ",0)


# In[202]:


df


# In[203]:


df["Gender"][628]


# In[204]:


df


# In[205]:


df["treatment"] =df['treatment'] .replace("No",0)
df["treatment"] =df['treatment'] .replace("Yes",1)


# In[206]:


df


# In[207]:


df.isnull().sum()


# In[208]:


df["family_history"] =df['family_history'] .replace("No",0)
df["family_history"] =df['family_history'] .replace("Yes",1)


# In[209]:


df


# In[210]:


df['mental_health_consequence'] =df['mental_health_consequence'] .replace("No",0)
df['mental_health_consequence'] =df['mental_health_consequence'] .replace("Yes",1)


# In[211]:


df


# In[212]:


df['phys_health_consequence'] =df['phys_health_consequence'] .replace("No",0)
df['phys_health_consequence'] =df['phys_health_consequence'] .replace("Yes",1)


# In[213]:


df


# In[214]:


df['obs_consequence'] =df['obs_consequence'] .replace("No",0)
df['obs_consequence'] =df['obs_consequence'] .replace("Yes",1)


# In[215]:


df


# # Dropping uwanted columns

# In[216]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
fig = plt.subplots(figsize= (10,10),dpi =200)
sn.set(font_scale =1.5)
sn.heatmap(df.corr(),square=True, cbar=True, annot=True)


# # Dropping unwanted columns

# In[217]:


df.drop(['Timestamp','Country','state','work_interfere','no_employees','leave','coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical','comments'],axis=1,inplace=True)


# In[218]:


df.drop(['remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity'],axis=1,inplace=True)


# In[219]:


df


# In[220]:


df.head()


# # Plotting with bar plot

# In[221]:


df['self_employed'].value_counts().plot(kind='bar',title='count of employed people')


# In[222]:


df['Gender'].value_counts().plot(kind='bar',title='Gender percentage')


# In[223]:


df['treatment'].value_counts().plot(kind='bar',title='Treatment')


# # Pie chart of observed problems

# In[224]:


plt.title('Observation of patient health problems ')
labels = ['No','Yes']
sizes = [len(df[df['obs_consequence'] == 0]),len(df[df['obs_consequence'] == 1])]
colors = ['blue', 'pink']
explode = (0.1, 0) 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# # Plotting of mental health consequences according to age

# In[225]:


plt.figure(figsize=(15,6))
sn.countplot(x='Age',data = df, hue = 'mental_health_consequence',palette='RdBu')
plt.show()


# # Splitting x and y

# In[226]:


X=df.drop(['self_employed'],axis=1)


# In[227]:


X.values


# In[228]:


y=df['self_employed']


# In[229]:


y.values


# In[230]:


from sklearn.model_selection import train_test_split


# In[231]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # KNeighbors Classifier 

# In[232]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 9)


# In[233]:


model.fit(X_train,y_train)
predict = model.predict(X_test)
predict


# In[234]:


model.score(X_test,y_test)*100


# In[235]:


from sklearn.grid_search import GridSearchCV


# In[236]:


k_range = list(range(1, 31))
print(k_range)


# In[237]:


param_grid = dict(n_neighbors=k_range)
print(param_grid)


# In[238]:


grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')


# In[239]:


grid.fit(X, y)


# In[240]:


grid.grid_scores_


# In[241]:


grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)


# In[242]:


plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[243]:


# Single best score achieved across all params (k)
print(grid.best_score_)


# In[244]:


# Dictionary containing the parameters (k) used to generate that score
print(grid.best_params_)


# # SVM 

# In[245]:


from sklearn.svm import SVC
model = SVC(random_state = 1)


# In[246]:


model.fit(X_train,y_train)
model.score(X_test,y_test)*100


# # Naive Bayes

# In[247]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[248]:


model.fit(X_train,y_train)
model.score(X_test,y_test)*100

