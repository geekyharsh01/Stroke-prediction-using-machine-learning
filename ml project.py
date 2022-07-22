#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


# In[77]:


df = pd.read_csv("/content/drive/MyDrive/healthcare-dataset-stroke-data.csv")
df1= pd.read_csv("/content/drive/MyDrive/healthcare-dataset-stroke-data.csv")
df2 = pd.read_csv("/content/drive/MyDrive/healthcare-dataset-stroke-data.csv")
df3 = pd.read_csv("/content/drive/MyDrive/healthcare-dataset-stroke-data.csv")
df


# In[78]:


#visualising the dataset


# In[78]:





# In[78]:





# In[78]:





# In[78]:





# In[78]:





# In[78]:





# In[79]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categ = ['gender','ever_married','work_type','Residence_type','smoking_status']
 
df[categ] = df[categ].apply(le.fit_transform)
# df1['class'] = encod.fit_transform(df1['class'])

df
 


# In[80]:


for i in df.columns:
  print (df[i].dtype)


# In[81]:


df.isnull().sum()


# In[82]:



df = df.dropna()
df= df.drop('id',axis=1)

df


# In[83]:


df1


# In[84]:


categ = ['gender','ever_married','work_type','Residence_type','smoking_status']
 
df1[categ] = df1[categ].apply(le.fit_transform)
# df1['class'] = encod.fit_transform(df1['class'])
df2[categ] = df2[categ].apply(le.fit_transform)

df3[categ] = df3[categ].apply(le.fit_transform)


df1
 


# In[85]:


df.describe()


# In[86]:


df1 = df1.dropna()

df1


# In[87]:


from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=4)
# df2['bmi']= imputer.fit_transform(df[['bmi']])
# # df2.isna().sum()
# df2.drop('id',axis=1,inplace=True)


# In[88]:


df2


# In[89]:


df3=df3.dropna()
df3.drop('id',axis=1,inplace = True)
df3


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


#df = normal with ffill and with id
# df1 is with all nan drop without id
#df2 is with without id anf ffill

x = df.drop('stroke',axis=1)
y = df['stroke']
x1 = df1.drop('stroke',axis=1)
y1 = df1['stroke']

x2 = df2.drop('stroke',axis=1)
y2= df2['stroke']

x3 = df3.drop('stroke',axis=1)
y3= df3['stroke']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)
x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1,test_size=0.2,random_state=20)
x_train2,x_test2,y_train2,y_test2 = train_test_split(x2,y2,test_size=0.2,random_state=20)
x_train3,x_test3,y_train3,y_test3 = train_test_split(x3,y3,test_size=0.2,random_state=20)


# In[92]:


# I will do everything for df only then i will copy the celll and do  copy past for df1 and df2 too and compare if there is any difference.

# The models i am goinf to train here are Decision Tree, logistic regressor, KNN, SVC, Random Forest


# In[170]:


# First we using Descision tree classifier
from sklearn.tree import DecisionTreeClassifier as dtc
model1 = dtc()


# In[171]:


model1.fit(x_train,y_train)


# In[95]:


y_pred = model1.predict(x_test)


# In[96]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[172]:


model1.score(x_test,y_test)


# In[98]:


import seaborn as sns
sns.distplot(y_test-y_pred)


# In[99]:


model1.fit(x_train1,y_train1)


# In[100]:


model1.score(x_test1,y_test1)


# In[101]:


# model1.fit(x_train2,y_train2)
# model1.score(x_test2,y_test2)


# In[102]:


model1.fit(x_train3,y_train3)
model1.score(x_test3,y_test3)


# In[103]:


#model2 = logistic regressor
from sklearn.linear_model import LogisticRegression


# In[167]:


model2 = LogisticRegression()


# In[168]:


model2.fit(x_train,y_train)


# In[169]:


model2.score(x_test,y_test)


# In[107]:


y_pred2 = model1.predict(x_test)
# from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred2))


# In[108]:


import seaborn as sns
sns.distplot(y_test-y_pred2)


# In[109]:


# model 3 = knn


# In[158]:


from sklearn.neighbors import KNeighborsClassifier as knn


# In[159]:


model3 = knn() 


# In[160]:


model3.fit(x_train,y_train)
model3.score(x_test,y_test)


# In[113]:


y_pred3 = model3.predict(x_test)


# In[114]:


print(classification_report(y_test,y_pred3))


# In[115]:


import seaborn as sns
sns.distplot(y_test-y_pred3)


# In[148]:


#model4 svc
from sklearn.svm import SVC
model4 = SVC(probability=True)


# In[152]:


model4.fit(x_train,y_train)
model4.score(x_test,y_test)


# In[118]:


y_pred4= model4.predict(x_test)
print(classification_report(y_test,y_pred4))


# In[119]:


import seaborn as sns
sns.distplot(y_test-y_pred4)


# In[119]:





# In[120]:


# model5 - Random Forest CLassifier
from sklearn.ensemble import RandomForestClassifier as rfc
model5 = rfc()


# In[121]:


model5.fit(x_train,y_train)
model5.score(x_test,y_test)


# In[122]:


y_pred5 = model5.predict(x_test)
print(classification_report(y_test,y_pred5))


# In[123]:


import seaborn as sns
sns.distplot(y_test-y_pred5)


# In[124]:


#plotinng ROC curve
y_prob5 = model5.predict_proba(x_test)
y_prob5 = y_prob5[:,1]
auc5 = roc_auc_score(y_test,y_prob5)
print("Roc score for model 5 is ",auc5)

fpr5,tpr5,thresh = roc_curve(y_test,y_prob5)


# In[125]:


plt.plot(fpr5,tpr5 , label = 'model5 AUCROC = %0.3f' %auc5)
plt.title('ROC plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[163]:


y_prob4 = model4.predict_proba(x_test)
y_prob4 = y_prob4[:,1]
auc4 = roc_auc_score(y_test,y_prob4)
print("Roc score for model 4 is ",auc4)

fpr4,tpr4,thresh = roc_curve(y_test,y_prob4)


# In[174]:


plt.plot(fpr4,tpr4 , label = 'model4 AUCROC = %0.3f' %auc4)
plt.title('ROC plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[164]:


y_prob3 = model3.predict_proba(x_test)
y_prob3 = y_prob3[:,1]
auc3 = roc_auc_score(y_test,y_prob3)
print("Roc score for model 3 is ",auc3)

fpr3,tpr3,thresh = roc_curve(y_test,y_prob3)


# In[165]:


plt.plot(fpr3,tpr3 , label = 'model3 AUCROC = %0.3f' %auc3)
plt.title('ROC plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[173]:


y_prob2 = model2.predict_proba(x_test)
y_prob2 = y_prob2[:,1]
auc2 = roc_auc_score(y_test,y_prob2)
print("Roc score for model 2 is ",auc2)

fpr2,tpr2,thresh = roc_curve(y_test,y_prob2)

plt.plot(fpr2,tpr2 , label = 'model2 AUCROC = %0.3f' %auc2)
plt.title('ROC plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[176]:


y_prob1 = model1.predict_proba(x_test)
y_prob1 = y_prob1[:,1]
auc1 = roc_auc_score(y_test,y_prob1)
print("Roc score for model 1 is ",auc1)

fpr1,tpr1,thresh = roc_curve(y_test,y_prob1)

plt.plot(fpr1,tpr1 , label = 'model1 AUCROC = %0.3f' %auc1)
plt.title('ROC plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[126]:


#Dimensionality Reduction
# PCA, LDA and SFS


# In[127]:


#1 - PCA 
from sklearn.decomposition  import PCA
pca = PCA(n_components= 5)
pca.fit(x)


# In[128]:


x_pca = pca.transform(x)
x_pca_t, x_pca_te = train_test_split(x_pca,test_size = 0.2)


# In[129]:


model5.fit(x_pca_t,y_train)
model5.score(x_pca_te,y_test)


# In[137]:


model4.fit(x_pca_t,y_train)
model4.score(x_pca_te,y_test)


# In[140]:


model3.fit(x_pca_t,y_train)
model3.score(x_pca_te,y_test)


# In[141]:


model2.fit(x_pca_t,y_train)
model2.score(x_pca_te,y_test)


# In[142]:


model1.fit(x_pca_t,y_train)
model1.score(x_pca_te,y_test)


# In[130]:


# 2- LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components= 1)
x_lda = lda.fit_transform(x_train,y_train)
x_lda_t = lda.fit_transform(x_test,y_test)


# In[131]:


model4.fit(x_lda,y_train)
model4.score(x_lda_t,y_test)


# In[131]:





# In[59]:


#Hyper paramaeter tuning 


# In[ ]:


#model1 - Decision tree classifier
param_dist = {
    'max_depth' : [1,2,3,4,5,6,7,8,9],
    'criterion' : ["gini","entropy"],
    'min_samples_split' : [1,2,3,4,5,6,7,9],
    'max_features' : ['auto','sqrt','log2'],
    'min_samples_leaf' : [1,2,3,4,5,6,7,8,0]


}

grid = GridSearchCV(model1,param_grid= param_dist,cv = 10, n_jobs = -1)
grid.fit(x_train,y_train)


# In[ ]:


grid.best_params_


# In[132]:


model1__ = dtc(criterion = 'gini', max_depth = 4,max_features = 'sqrt',min_samples_leaf = 1, min_samples_split= 7)
model1__.fit(x_train,y_train)
model1__.score(x_test,y_test)


# In[177]:


y_pred_dt = model1__.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt))


# In[ ]:


# hyparameter tuning model2 
param_dist = {
   'penalty' : ['l1','l2','none','elasticnet'],
   'dual': [True,False],
   'C' : [0.5,0.7,1,1.5,2,3,4,5],
   'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']


}

grid1 = GridSearchCV(model2,param_grid= param_dist,cv = 10, n_jobs = -1)
grid1.fit(x_train,y_train)


# In[ ]:


grid1.best_params_


# In[151]:


#implementing in model

model2__ = LogisticRegression( C= 0.5, dual= False, penalty='l2', solver =  'newton-cg')
model2__.fit(x_train,y_train)
model2__.score(x_test,y_test)


# In[178]:


y_pred_dt = model2__.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt))


# In[60]:


#hyperparameter tuning of model3
param_dist = {
   'n_neighbors' : [2,4,6,8,10],
   'weights': ['uniform', 'distance'],
   'algorithm' : ['auto','ball_tree','kd_tree','brute'],
   'leaf_size' : [10,20,30,50,70]
   



}

grid2 = GridSearchCV(model3,param_grid= param_dist,cv = 10, n_jobs = -1)
grid2.fit(x_train,y_train)


# In[63]:


grid2.best_params_


# In[150]:


# creating the model and finding the accuracy
from sklearn.neighbors import KNeighborsClassifier as knn
model3__ = knn(algorithm = 'auto', leaf_size =  10, n_neighbors = 10, weights = 'uniform') 
model3__.fit(x_train,y_train)
model3__.score(x_test,y_test)


# In[179]:


y_pred_dt = model3__.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt))


# In[ ]:





# In[64]:


# hypertuning model 4 that is SVC
param_dist = {
    'C' : [0.5,1,1.5,2,3,5],
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'gamma' : ['scale','auto'],
    'shrinking' : [True,False]
    
   


}

grid3 = GridSearchCV(model4,param_grid= param_dist,cv = 10, n_jobs = -1)
grid3.fit(x_train,y_train)


# In[ ]:


grid3.best_params_


# In[ ]:


# creating hypertuned model model4


# In[ ]:


# now hypertunoing model 5 that is Random forest
param_dist = {
    'n_estimators' : [100,200,400,600,800],
   
    
    'min_samples_split' : [2,3,5,7,9,11],
    
    'max_features' : ['auto','sqrt','log2'],
   
    
   
  


}

grid4 = GridSearchCV(model5,param_grid= param_dist,cv = 10, n_jobs = -1)
grid4.fit(x_train,y_train)

grid4.best_params_


# In[149]:


# training the model with tuned hyperparameters
from sklearn.ensemble import RandomForestClassifier as rfc
model5__ = rfc(max_features='sqrt', min_samples_split= 11, n_estimators = 100)
model5__.fit(x_train,y_train)
model5__.score(x_test,y_test)


# In[180]:


y_pred_dt = model5__.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt))


# In[ ]:




