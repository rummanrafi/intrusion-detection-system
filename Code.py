#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Declaration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score

warnings.filterwarnings('ignore')

#Get_data
df=pd.read_csv('D:\\MS in US\\MS in Computer Science\\Spring Semester 2021\\Thesis\\Datasets\\UNSW-NB15.csv')

#Divide data
X=df.iloc[:,0:38]
y=df['Label']

#Split data
seed=42
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=seed)

#K_Nearest_Neighbor
model_KNN=KNeighborsClassifier(n_neighbors=6, weights='uniform', p=2, metric='minkowski')
model_KNN.fit(X_train,y_train)
y_test_pred_KNN=model_KNN.predict(X_test)
acc_KNN=accuracy_score(y_test,y_test_pred_KNN)
print(confusion_matrix(y_test,y_test_pred_KNN))
print(classification_report(y_test,y_test_pred_KNN))
y_score_KNN=model_KNN.predict_proba(X_test)[:,1]
fpr, tpr,thresholds=roc_curve(y_test,y_score_KNN)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC CURVE For KNN')
plt.show()
roc_auc_score_KNN=roc_auc_score(y_test,y_score_KNN)
print('ROC_AUC_Score of KNN  Model : ' , roc_auc_score_KNN )
print('Accuracy of KNN  Model : ' , acc_KNN )


#LogisticRegression
model_LR=LogisticRegression(random_state=seed)
model_LR.fit(X_train,y_train)
y_test_pred_LR=model_LR.predict(X_test)
acc_LR=accuracy_score(y_test,y_test_pred_LR)
print(confusion_matrix(y_test,y_test_pred_LR))
print(classification_report(y_test,y_test_pred_LR))
y_score_LR=model_LR.predict_proba(X_test)[:,1]
fpr, tpr,thresholds=roc_curve(y_test,y_score_LR)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC CURVE For LogisticRegression')
plt.show()
roc_auc_score_LR=roc_auc_score(y_test,y_score_LR)
print('ROC_AUC_Score of LogisticRegression Model : ' , roc_auc_score_LR )
print('Accuracy of LogisticRegression Model : ' , acc_LR )

#RandomForest
model_RF=RandomForestClassifier(n_estimators=400,
                               min_samples_leaf=0.12,
                               random_state=seed)
model_RF.fit(X_train,y_train)
y_test_pred_RF=model_RF.predict(X_test)
acc_RF=accuracy_score(y_test,y_test_pred_RF)
print(confusion_matrix(y_test,y_test_pred_RF))
print(classification_report(y_test,y_test_pred_RF))
y_score_RF=model_RF.predict_proba(X_test)[:,1]
fpr, tpr,thresholds=roc_curve(y_test,y_score_RF)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC CURVE For RandomForest')
plt.show()
roc_auc_score_RF=roc_auc_score(y_test,y_score_RF)
print('ROC_AUC_Score of RandomForest Model : ' , roc_auc_score_RF )
print('Accuracy of RandomForest Model : ' , acc_RF )



#Accuracy_Score
print('Accuracy of KNN  Model                    : ' , acc_KNN )
print('Accuracy of Logistic Regression Model     : ' , acc_LR )
print('Accuracy of Random Forest Classifier Model: ' , acc_RF )





