
# coding: utf-8

# 参考书和网站：
# 
# [python for data minging](http://guidetodatamining.com/) 
# 
# [我爱机器学习](http://www.52ml.net/)

# #模型训练使用流程
# - 数据抽取
# - 数据探索
# - 建模数据准备
# - 变量选择
# - 模型开发与验证
# - 模型部署
# - 模型监督

# In[86]:

import os
os.chdir(r'D:\code\CASE2\Python')

# ## 步骤一：构造营销响应模型

# ###  1、	数据获取与导入的S（抽样）阶段。

# - 规整数据集

# In[87]:


import pandas as pd
model_data = pd.read_csv("donations2.csv").drop(["ID","TARGET_D"],1)
model_data.head()


# In[88]:


model_data.dtypes


# In[89]:


#model_data["TARGET_B"]=pd.Categorical(model_data["TARGET_B"],ordered=False)
model_data["StatusCat96NK"]=pd.Categorical(model_data["StatusCat96NK"],ordered=False)
model_data["DemCluster"]=pd.Categorical(model_data["DemCluster"],ordered=False)
model_data["DemGender"]=pd.Categorical(model_data["DemGender"],ordered=False)
model_data["DemHomeOwner"]=pd.Categorical(model_data["DemHomeOwner"],ordered=False)


# 在pandas中的官方在线文档中，给出了pandas因子变量的详细论述，并在适当位置与R语言进行了对比描述。
# http://pandas.pydata.org/pandas-docs/stable/categorical.html#working-with-categories

# In[90]:


model_data.dtypes


# In[91]:


y = 'TARGET_B'
var_c = ["GiftCnt36","GiftCntAll","GiftCntCard36","GiftCntCardAll","GiftTimeLast","GiftTimeFirst",         "PromCnt12","PromCnt36","PromCntAll","PromCntCard12","PromCntCard36","PromCntCardAll",         "StatusCatStarAll","DemAge","DemMedHomeValue","DemPctVeterans","DemMedIncome","GiftAvgLast",         "GiftAvg36","GiftAvgAll","GiftAvgCard36"]
var_d = list(set(model_data.columns)-set(var_c)-set([y]))


# In[92]:


X = model_data[var_c+var_d]
Y = model_data[y]


# - 筛选预测能力强的变量

# In[93]:


from woe import WoE


# **WoE类参数说明**:
# + **qnt_num**:int,等频分箱个数,默认16
# + **min_block_size**:int,最小观测数目，默认16
# + **spec_values**:dict,若为分类自变量，指派替换值
# + **v_type**:str,自变量类型,分类:‘d’,连续变量:‘c’，默认'c'
# + **bins**:list,预定义的连续变量分箱区间
# + **t_type**:str,目标变量类型,二分类:‘b’,连续变量:‘c’，默认'b'

# **WoE类重要方法**:
# 
# + **plot**:绘制WOE图
# + **transform**:转换数据为WOE数据
# + **fit_transform**:转换数据为WOE数据
# + **optimize**:连续变量使用最优分箱

# **WoE类重要属性**:
# + **bins**:分箱结果汇总
# + **iv**:变量的信息价值

# + 根据IV值筛选变量-分类变量

# In[94]:


from woe import WoE
iv_d = {}
for i in var_d:
    iv_d[i] = WoE(v_type='d',t_type='b').fit(X[i],Y).iv

pd.Series(iv_d).sort_values(ascending=False)


# In[95]:


var_d_s = list(set(var_d)-set(["DemHomeOwner","DemGender"]))


# + 根据IV值筛选变量-连续变量

# In[96]:


iv_c = {}
for i in var_c:
    iv_c[i] = WoE(v_type='c',t_type='b',qnt_num=3).fit(X[i],Y).iv 

pd.Series(iv_c).sort_values(ascending=False)


# In[97]:


var_c_s = list(set(var_c)-set(["PromCntCard12","PromCnt12","DemMedHomeValue","PromCnt36", "DemAge","DemPctVeterans","DemMedIncome","StatusCatStarAll","GiftCntCard36"]))


# In[98]:


X = model_data[var_c_s+var_d_s]
Y = model_data[y]


# ### 2、针对每个变量的E（探索）阶段

# - 对连续变量的统计探索

# In[99]:


X[var_c_s].describe().T


# In[100]:


import matplotlib.pyplot as plt
plt.hist(X["PromCntAll"], bins=20)  
plt.show()


# In[101]:


abs((X[var_c_s].mode().ix[0,]-X[var_c_s].median())/(X[var_c_s].quantile(0.75)-X[var_c_s].quantile(0.25)))


# - 对分类变量的统计探索

# In[102]:


X["DemCluster"].value_counts()


# ### 3、针对有问题的变量进行修改的M（修改）阶段

# - 将连续变量的错误值改为缺失值

# In[103]:


X.isnull().sum()/(X.count()+X.isnull().sum())


# - 将连续变量的缺失值用中位数填补

# In[104]:


top_state = X.GiftAvgCard36.median()
X.GiftAvgCard36.fillna(value=top_state, inplace=True)


# - 对分类水平过多的变量进行合并（或概化）

# In[105]:


X.DemCluster.value_counts()


# In[106]:


X_rep=X.replace({"DemCluster":(37,48,5 ,47,7 ,29,32,9 ,50,6,33)},100)


# In[107]:


for i in var_d_s:
    X_rep[i] = WoE(v_type='d').fit_transform(X[i],Y)


# In[108]:


import sklearn.ensemble as ensemble
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
rfc_model = rfc.fit(X_rep, Y)
rfc_model.feature_importances_
rfc_fi = pd.DataFrame()
rfc_fi["features"] = list(X.columns)
rfc_fi["importance"] = list(rfc_model.feature_importances_)
rfc_fi=rfc_fi.set_index("features",drop=True)
rfc_fi.sort_index(by="importance",ascending=False).plot(kind="bar")


# In[109]:


var_x = ["GiftAvgAll","DemCluster","PromCntAll","GiftTimeFirst","GiftAvg36","GiftTimeLast","GiftAvgCard36","GiftCntAll","PromCntCardAll","GiftAvgLast","PromCntCard36","GiftCntCardAll",         "GiftCnt36"]


# - 3）解释变量分布转换

# In[25]:


import matplotlib.pyplot as plt
for i in var_x:
    plt.hist(X_rep[i], bins=20)
    plt.show()


# In[110]:


skew_var_x = {}
for i in var_x:
    skew_var_x[i]=abs(X_rep[i].skew())
    
skew=pd.Series(skew_var_x).sort_values(ascending=False)
skew


# In[111]:


var_x_ln = skew.index[skew>=1]
import numpy as np
for i in var_x_ln:
    if min(X_rep[i])<0:
        X_rep[i] =np.log(X_rep[i]+abs(min(X_rep[i]))+0.01)
    else:
        X_rep[i] =np.log(X_rep[i]+0.01)


# In[112]:


skew_var_x = {}
for i in var_x:
    skew_var_x[i]=abs(X_rep[i].skew())
    skew=pd.Series(skew_var_x).sort_values(ascending=False)
skew


# - 3）变量压缩

# In[146]:
#%%
from VarSelec import *
#%%
X_rep_reduc=Var_Select_auto(X_rep)
X_rep_reduc.head()


# In[155]:


X_rep_reduc.corr()


# ### 4、建立逻辑回归模型M（建模）阶段

# - 分成训练集和测试集，比例为6:4

# In[147]:


import sklearn.cross_validation as cross_validation
train_data, test_data, train_target, test_target = cross_validation.train_test_split(X_rep_reduc, Y, test_size=0.3, random_state=0)


# - 模型训练

# - 使用全部变量进行logistic回归

# In[148]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

train_data = min_max_scaler.fit_transform(train_data)
test_data = min_max_scaler.fit_transform(test_data)
train_data


# In[149]:


import sklearn.linear_model as linear_model
logistic_model = linear_model.LogisticRegression(class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l1', random_state=None, tol=0.001)


# In[150]:


from sklearn.model_selection import ParameterGrid, GridSearchCV

C=np.logspace(-3,0,20,base=10)

param_grid = {'C': C}

clf_cv = GridSearchCV(estimator=logistic_model, 
                      param_grid=param_grid, 
                      cv=5, 
                      scoring='roc_auc')

clf_cv.fit(train_data, train_target)


# In[151]:


import sklearn.linear_model as linear_model
logistic_model = linear_model.LogisticRegression(C=clf_cv.best_params_["C"], class_weight=None,                                                  dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l1', random_state=None, tol=0.001)
logistic_model.fit(train_data, train_target)

#%%
# ### 5、模型验证A（验证）阶段。

# - 对逻辑回归模型进行评估

# In[56]:


test_est = logistic_model.predict(test_data)
train_est = logistic_model.predict(train_data)


# In[57]:


test_est_p = logistic_model.predict_proba(test_data)[:,1]
train_est_p = logistic_model.predict_proba(train_data)[:,1]


# In[58]:


import sklearn.metrics as metrics
print(metrics.classification_report(test_target, test_est))


# In[59]:


print(metrics.classification_report(train_target, train_est))


# In[60]:
# - 目标样本和非目标样本的分数分布

# In[65]:


import seaborn as sns
red, blue = sns.color_palette("Set1",2)


# In[66]:


sns.kdeplot(test_est_p[test_target==1], shade=True, color=red)
sns.kdeplot(test_est_p[test_target==0], shade=True, color=blue)


# - ROC曲线

# In[67]:


fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color=blue)
plt.plot(fpr_train, tpr_train, color=red)
plt.title('ROC curve')
print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test))


# In[68]:


test_x_axis = np.arange(len(fpr_test))/float(len(fpr_test))
train_x_axis = np.arange(len(fpr_train))/float(len(fpr_train))
plt.figure(figsize=[6,6])
plt.plot(fpr_test, test_x_axis, color=blue)
plt.plot(tpr_test, test_x_axis, color=blue)
plt.plot(fpr_train, train_x_axis, color=red)
plt.plot(tpr_train, train_x_axis, color=red)
plt.title('KS curve')


# - 构建神经网络并评估

# In[69]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[70]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='logistic', alpha=0.1, max_iter=1000)


# In[71]:


from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {
    'hidden_layer_sizes':[(10, ), (15, ), (20, ), (5, 5)],
    'activation':['logistic', 'tanh', 'relu'], 
    'alpha':[0.001, 0.01, 0.1, 0.2, 0.4, 1, 10]
}
mlp = MLPClassifier(max_iter=1000)
gcv = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                   scoring='roc_auc', cv=4, n_jobs=-1)
gcv.fit(scaled_train_data, train_target)


# In[72]:


gcv.best_params_

#{'activation': 'tanh', 'alpha': 0.4, 'hidden_layer_sizes': (20,)}
# In[73]:


mlp = MLPClassifier(hidden_layer_sizes=gcv.best_params_["hidden_layer_sizes"], 
                    activation=gcv.best_params_["activation"], alpha=gcv.best_params_["alpha"], max_iter=1000)

mlp.fit(scaled_train_data, train_target)


# In[74]:


train_predict = mlp.predict(scaled_train_data)
test_predict = mlp.predict(scaled_test_data)


# In[75]:


train_proba = mlp.predict_proba(scaled_train_data)[:, 1]  
test_proba = mlp.predict_proba(scaled_test_data)[:, 1]


# In[76]:


from sklearn import metrics

print(metrics.confusion_matrix(test_target, test_predict, labels=[0, 1]))
print(metrics.classification_report(test_target, test_predict))


# In[77]:


fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_proba)

plt.figure(figsize=[4, 4])
plt.plot(fpr_test, tpr_test, 'b-')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test))


# ### 模型永久化

# In[78]:


import pickle as pickle
model_file = open(r'logitic.model', 'wb')
pickle.dump(logistic_model, model_file)
model_file.close()


# In[79]:


model_load_file = open(r'logitic.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()


# In[80]:


test_est_load = model_load.predict(test_data)


# In[81]:


pd.crosstab(test_est_load,test_est)
#%%

