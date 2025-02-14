#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


diabetes = pd.read_csv('diabetes.csv')


# In[7]:


diabetes.head()


# In[8]:


diabetes.info()


# In[9]:


diabetes.columns


# In[10]:


diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)


# In[11]:


diabetes.head()


# In[12]:


diabetes.fillna(diabetes.mean(), inplace=True)


# In[13]:


diabetes.head()


# In[14]:


sns.countplot(x='Outcome', data=diabetes, palette='pastel')


# In[15]:


sns.boxplot(x='Outcome', data=diabetes, y='Age',palette='pastel')


# In[16]:


sns.pairplot(data=diabetes, hue='Outcome', vars=['Glucose', 'BMI', 'Age', 'BloodPressure'], palette='pastel')
plt.show()


# In[17]:


X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


smote = SMOTE(random_state=101)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[23]:


X_train_resampled = scaler.fit_transform(X_train_resampled)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


logmodel = LogisticRegression(random_state=101, max_iter=1000)
logmodel.fit(X_train, y_train)

# Train Logistic Regression on SMOTE data
logmodel_smote = LogisticRegression(random_state=101, max_iter=1000)
logmodel_smote.fit(X_train_resampled, y_train_resampled)

# Make predictions using both models
predictions_log = logmodel.predict(X_test)
predictions_smote = logmodel_smote.predict(X_test)

# Evaluate Logistic Regression (Original)
print("Logistic Regression (Original Data):")
print("Accuracy:", accuracy_score(y_test, predictions_log))
print(confusion_matrix(y_test, predictions_log))
print(classification_report(y_test, predictions_log))

# Evaluate Logistic Regression (SMOTE)
print("\nLogistic Regression (After SMOTE):")
print("Accuracy:", accuracy_score(y_test, predictions_smote))
print(confusion_matrix(y_test, predictions_smote))
print(classification_report(y_test, predictions_smote))


# In[56]:


# Function to plot confusion matrix
def plot_confusion_matrix(ax, y_true, predictions_cm, title, cmap='Greens'):
    cm = confusion_matrix(y_true, predictions_cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion_matrix(axes[0], y_test, predictions_log, "Logistic Regression (Original)")
plot_confusion_matrix(axes[1], y_test, predictions_smote, "Logistic Regression (SMOTE)")

plt.tight_layout()
plt.show()


# In[27]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


# In[28]:


knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)


predictions_knn = knn.predict(X_test)


# In[29]:


print(classification_report(y_test, predictions_knn))

print(accuracy_score(y_test,predictions_knn))

print(confusion_matrix(y_test,predictions_knn))


# In[31]:


from sklearn.svm import SVC

svm_model = SVC(kernel='linear', random_state=101)
svm_model.fit(X_train_resampled, y_train_resampled)


# In[34]:


predictions_log = logmodel.predict(X_test)
predictions_knn = knn.predict(X_test)
predictions_svm = svm_model.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression:")
print(confusion_matrix(y_test, predictions_log))
print(classification_report(y_test, predictions_log))

# Evaluate KNN
print("knn")
print(confusion_matrix(y_test, predictions_knn))
print(classification_report(y_test, predictions_knn))

# Evaluate SVM
print("svm")
print(confusion_matrix(y_test, predictions_svm))
print(classification_report(y_test, predictions_svm))
print("smote")
print(confusion_matrix(y_test, predictions_smote))
print(classification_report(y_test, predictions_smote))


# In[52]:


fig, axes = plt.subplots(1, 4, figsize=(15, 5))

plot_confusion_matrix(axes[0], y_test, predictions_log, "logistic regression", cmap='Greens')
plot_confusion_matrix(axes[1], y_test, predictions_knn, "knn", cmap='Blues')
plot_confusion_matrix(axes[2], y_test, predictions_svm, "svm", cmap='Reds')
plot_confusion_matrix(axes[3], y_test, predictions_smote, "smote", cmap='Oranges')

plt.tight_layout()
plt.show()


# In[ ]:




