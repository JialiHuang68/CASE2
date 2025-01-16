
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#步骤二：潜在客户价值挖掘模型" data-toc-modified-id="步骤二：潜在客户价值挖掘模型-0.1">步骤二：潜在客户价值挖掘模型</a></span><ul class="toc-item"><li><span><a href="#1、----数据获取与导入" data-toc-modified-id="1、----数据获取与导入-0.1.1">1、    数据获取与导入</a></span></li><li><span><a href="#2、筛选预测能力强的变量" data-toc-modified-id="2、筛选预测能力强的变量-0.1.2">2、筛选预测能力强的变量</a></span></li><li><span><a href="#3、对进行探索性分析" data-toc-modified-id="3、对进行探索性分析-0.1.3">3、对进行探索性分析</a></span></li><li><span><a href="#4、针对有问题的变量进行修改" data-toc-modified-id="4、针对有问题的变量进行修改-0.1.4">4、针对有问题的变量进行修改</a></span></li><li><span><a href="#5、前向选择法筛选变量" data-toc-modified-id="5、前向选择法筛选变量-0.1.5">5、前向选择法筛选变量</a></span></li><li><span><a href="#6、建立线性回归模型" data-toc-modified-id="6、建立线性回归模型-0.1.6">6、建立线性回归模型</a></span></li></ul></li></ul></li></ul></div>

# In[1]:


import pandas as pd
import numpy as np
import os
os.chdir(r'D:\code\CASE2\Python')


# In[2]:


#os.chdir(r"C:\Users\boccfc\Desktop\慈善捐款精准营销的两阶段预测模型")


# ## 步骤二：潜在客户价值挖掘模型

# ###  1、	数据获取与导入

# - 规整数据集

# In[3]:


model_data = pd.read_csv("donations2.csv").drop(["ID","TARGET_B"],1)
model_data.head()


# In[4]:


model_data.dtypes


# In[5]:


#model_data["TARGET_B"]=pd.Categorical(model_data["TARGET_B"],ordered=False)
model_data["StatusCat96NK"]=pd.Categorical(model_data["StatusCat96NK"],ordered=False)
model_data["DemCluster"]=pd.Categorical(model_data["DemCluster"],ordered=False)
model_data["DemGender"]=pd.Categorical(model_data["DemGender"],ordered=False)
model_data["DemHomeOwner"]=pd.Categorical(model_data["DemHomeOwner"],ordered=False)


# 在pandas中的官方在线文档中，给出了pandas因子变量的详细论述，并在适当位置与R语言进行了对比描述。
# http://pandas.pydata.org/pandas-docs/stable/categorical.html#working-with-categories

# In[6]:


model_data.dtypes[-10:-1]


# In[20]:


y = ["TARGET_D"]
var_c = ["GiftCnt36","GiftCntAll","GiftCntCard36","GiftCntCardAll","GiftTimeLast",
         "GiftTimeFirst","PromCnt12","PromCnt36","PromCntAll","PromCntCard12",
         "PromCntCard36","PromCntCardAll","StatusCatStarAll","DemAge",
         "DemMedHomeValue","DemPctVeterans","DemMedIncome","GiftAvgLast",
         "GiftAvg36","GiftAvgAll","GiftAvgCard36"]
var_d = list(set(model_data.columns)-set(var_c)-set(y))


# In[21]:


X = model_data[var_c+var_d]
Y = model_data[y]


# ###  2、筛选预测能力强的变量

# + 连续变量筛选-相关性

# In[39]:


corr_s = abs(model_data[y+var_c].corr(method = 'spearman'))
corr_s = pd.DataFrame(corr_s.iloc[0,:])

corr_p = abs(model_data[y+var_c].corr(method = 'pearson'))
corr_p = pd.DataFrame(corr_p.iloc[0,:])

corr_sp = pd.concat([corr_s,corr_p],axis = 1)
corr_sp.columns = ['spearman','pearson']


# In[40]:


corr_sp[(corr_sp['spearman'] <= 0.1) & (corr_sp['pearson'] <= 0.1)]


# In[42]:


var_c_s = set(var_c) - set(['PromCnt12','PromCnt36','PromCntCard12','DemAge','DemPctVeterans','DemMedIncome'])
var_c_s = list(var_c_s)
var_c_s


# + 分类变量筛选-方差分析

# In[43]:


var_d


# In[44]:


import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols
for i in var_d:
    formula = "TARGET_D ~" + str(i)
    print(anova.anova_lm(ols(formula,data = model_data[var_d+['TARGET_D']]).fit()))


# In[45]:


var_d_s = list(set(var_d) - set(["DemHomeOwner"]))


# ### 3、对进行探索性分析

# - 对连续变量的统计探索

# In[46]:


X = model_data[var_c_s+var_d_s]
Y = model_data[y]


# In[47]:


model_data[var_c_s+var_d_s].head()


# In[48]:


# 描述性统计分析
X[var_c_s].describe().T


# In[49]:


# 数据分布探索
#import matplotlib.pyplot as plt
#for i in var_c_s:    
#    plt.hist(X[i], bins=50,label = i)  
#    plt.title(i)
#    plt.show()


# In[50]:


abs((X[var_c_s].mode().ix[0,]-X[var_c_s].median())/(X[var_c_s].quantile(0.75)-X[var_c_s].quantile(0.25)))


# - 对分类变量的统计探索

# In[51]:


X["DemCluster"].value_counts()[:10]


# ### 4、针对有问题的变量进行修改

# - 连续变量：将变量中错误值替换为缺失值，然后进行缺失值填补

# In[52]:


X['DemMedHomeValue'].replace(0,np.nan,inplace = True)


# In[53]:


X.isnull().sum()/(X.count()+X.isnull().sum())


# In[54]:


X["GiftAvgCard36"].fillna(X["GiftAvgCard36"].median(),inplace = True)
X["DemMedHomeValue"].fillna(X["DemMedHomeValue"].median(),inplace = True)


# - 解释变量分布转换

# In[55]:


import matplotlib.pyplot as plt
for i in var_c_s:
    plt.hist(X[i], bins=20)
    plt.show()


# In[56]:


skew_var_x = {}
for i in var_c_s:
    skew_var_x[i]=abs(X[i].skew())
    
skew=pd.Series(skew_var_x).sort_values(ascending=False)
skew


# In[57]:


var_x_ln = skew.index[skew>=1]
import numpy as np
for i in var_x_ln:
    if min(X[i])<0:
        X[i] =np.log(X[i]+abs(min(X[i]))+0.01)
    else:
        X[i] =np.log(X[i]+0.01)


# In[58]:


skew_var_x = {}
for i in var_c_s:
    skew_var_x[i]=abs(X[i].skew())
    skew=pd.Series(skew_var_x).sort_values(ascending=False)
skew


# - 对分类水平过多的变量进行合并（或概化）

# In[69]:


X.DemCluster.value_counts()[:15]


# In[70]:


X_rep = X
X_rep["DemCluster"]=X["DemCluster"].replace([37,48,5 ,47,7 ,29,32,9 ,50,6,33,4,19,52],100)


# - 水平数过多的分类变量进行水平合并

# In[71]:


# 统计每个水平的对应目标变量的均值，和每个水平数量
StatusCat96NK_class = model_data[['StatusCat96NK','TARGET_D']]                .groupby('StatusCat96NK', as_index = False)['TARGET_D']                .agg({'mean' : 'mean', 'count':'count'})

# 分析各个水平的相近程度
StatusCat96NK_class['ratio'] = StatusCat96NK_class["mean"]                .map(lambda x: x/StatusCat96NK_class["mean"].max())
# 根据相近程度和每个水平的个数，进行水平合并
StatusCat96NK_class["StatusCat96NK_new"] = StatusCat96NK_class['ratio']                .map(lambda x: 1 if x > 0.5 else 0)
# 将合并前水平与合并后水平进行字典映射    
StatusCat96NK_dict = {}
for i in range(len(StatusCat96NK_class)):
    StatusCat96NK_dict[StatusCat96NK_class.iloc[i,0]] = StatusCat96NK_class.iloc[i,-1]

# 利用上步字典对变量进行重编码
X['StatusCat96NK_new'] =  X['StatusCat96NK'].map(StatusCat96NK_dict)  
print(StatusCat96NK_class)


# In[72]:


# 统计每个水平的对应目标变量的均值，和每个水平数量
DemCluster_class = model_data[['DemCluster','TARGET_D']].groupby('DemCluster', as_index = False)['TARGET_D'].agg({'mean' : 'mean', 'count':'count'})

# 分析各个水平的相近程度
DemCluster_class['ratio'] = DemCluster_class["mean"].map(lambda x: x/DemCluster_class["mean"].max())
# 根据相近程度和每个水平的个数，进行水平合并
DemCluster_class["DemCluster_new"] = DemCluster_class['ratio'].map(lambda x: 1 if x > 0.75 else 0)
# 将合并前水平与合并后水平进行字典映射    
DemCluster_dict = {}
for i in range(len(DemCluster_class)):
    DemCluster_dict[DemCluster_class.iloc[i,0]] = DemCluster_class.iloc[i,-1]

# 利用上步字典对变量进行重编码
X['DemCluster_new'] =  X['DemCluster'].map(DemCluster_dict) 


# In[73]:


DemCluster_class[DemCluster_class.DemCluster_new == 1]['count'].sum()/DemCluster_class['count'].sum()


# In[74]:


var_d_s = ['DemGender', 'StatusCat96NK_new', 'DemCluster_new']


# In[75]:
# ###  5、前向选择法筛选变量

# In[100]:


import statsmodels.formula.api as smf
def forward_selected(data, response):
    """
    Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return selected


# In[101]:


forward_selected(pd.concat([X[var_c_s+var_d_s],Y],axis = 1), 'TARGET_D')


# In[86]:


var_c_s = ['GiftAvgAll',
 'GiftAvgLast', 'GiftAvg36',
 'GiftCntAll', 'GiftAvgCard36',
 'GiftCntCard36', 'PromCntAll',
 'StatusCatStarAll']

var_d_s=['DemCluster_new', 'DemGender']


# In[87]:


model_final = pd.concat([X[var_d_s + var_c_s],Y],axis = 1)
model_final.columns

#%%
model_final1=model_final[model_final.TARGET_D.isnull()==False]
model_final1 = model_final1.fillna(model_final1.median())
# ### 6、建立线性回归模型

# - 使用筛选出的变量进行线性回归

# In[88]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

# fit our model with .fit() and show results
# we use statsmodels' formula API to invoke the syntax below,
# where we write out the formula using ~
formula = 'TARGET_D ~' + '+'.join(var_c_s)
    
donation_model = ols(formula,data=model_final1).fit()
# summarize our model
print(donation_model.summary())

#%%
