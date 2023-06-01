import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_3.csv")
print(data.shape, '\n', data.head(5))

random_seed=1234

# 划分特征和标签
x=data.drop(['fraud'], axis=1)
y=data.fraud
print(x.shape, y.shape)

# # 归一化
# mm=MinMaxScaler()
# X=pd.DataFrame(mm.fit_transform(x))
# X.columns=x.columns
# print(X.head(5), X.shape)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

######Lasso特征选择
#调用LassoCV函数，并进行交叉验证，默认cv=3
lasso_model = LassoCV(alphas = [0.1,1,0.001,0.0005],random_state=random_seed).fit(X_train,y_train)
print(lasso_model.alpha_) #模型所选择的最优正则化参数alpha

#输出看模型最终选择了几个特征向量，剔除了几个特征向量
coef = pd.Series(lasso_model.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#
# 索引和重要性做成dataframe形式
FI_lasso = pd.DataFrame({"Feature Importance":lasso_model.coef_}, index=x.columns)
## 由高到低进行排序
FI_lasso.sort_values("Feature Importance",ascending=False).round(3)
# FI_lasso.to_csv('./FI.csv')
# 获取重要程度大于0的系数指标
FI_lasso[FI_lasso["Feature Importance"] !=0 ].sort_values("Feature Importance").plot(kind="barh",color='cornflowerblue',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature coefficient',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.6.jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()

# Lasso特征选择后的 训练集 和 测试集
drop_colums=coef.index[abs(coef.values)==0]
X_train_lasso=X_train.drop(drop_colums, axis=1)
X_test_lasso=X_test.drop(drop_colums, axis=1)
print('Lasso特征选择后的训练集和特征集的维度', X_train_lasso.shape, X_test_lasso.shape)

# SMOTE采样
X_train_lasso_SMOTE, y_train_lasso_SMOTE =SMOTE(random_state=random_seed).fit_resample(X_train_lasso, y_train)
print('SMOTE_train_set:', Counter(y_train_lasso_SMOTE), '\n', 'test_set:', Counter(y_test))

# BorderlineSMOTE采样
X_train_lasso_B_SMOTE, y_train_lasso_B_SMOTE =BorderlineSMOTE(random_state=random_seed).fit_resample(X_train_lasso, y_train)
print('B_SMOTE_train_set:', Counter(y_train_lasso_B_SMOTE), '\n', 'test_set:', Counter(y_test))

# SMOTE+Tomek_Link采样
X_train_lasso_S_T_L, y_train_lasso_S_T_L =SMOTETomek(random_state=random_seed).fit_resample(X_train_lasso, y_train)
print('S_T_L_train_set:', Counter(y_train_lasso_S_T_L), '\n', 'test_set:', Counter(y_test))

# # 经过比较后，Lasso + SMOTE_Tomek_link 的预处理组合性能最好，因此将训练集和测试集保存
# data_train=pd.concat([X_train_lasso_S_T_L, y_train_lasso_S_T_L], axis=1)
# data_train=data_train.reset_index(drop=True)
# data_test=pd.concat([X_test_lasso, y_test], axis=1)
# data_test=data_test.reset_index(drop=True)
# print(data_train.shape, data_test.shape)
#
# data_train.to_csv(path_or_buf=r'data/data_train.csv', index=None)
# data_test.to_csv(path_or_buf=r'data/data_test.csv', index=None)

#####互信息分类
k_best = 12
mic_model=SelectKBest(MIC, k=k_best)
X_mic = mic_model.fit_transform(X_train, y_train)
mic_scores=mic_model.scores_
mic_indices=np.argsort(mic_scores)[::-1]
mic_k_best_features = list(X_train.columns.values[mic_indices[0:k_best]])
FI_mic = pd.DataFrame({"Feature Importance":mic_scores}, index=X_train.columns)
FI_mic[FI_mic["Feature Importance"] !=0 ].sort_values("Feature Importance").plot(kind="barh",color='firebrick',alpha=0.8)
plt.xticks(rotation=0,fontsize=11)
plt.xlabel('特征重要程度',fontsize=11)
plt.ylabel('特征名称',fontsize=11)
plt.show()

# MIC特征选择后的 训练集 和 测试集
X_train_MIC=X_train[mic_k_best_features]
X_test_MIC=X_test[mic_k_best_features]
print('MIC特征选择后的训练集和特征集的维度', X_train_MIC.shape, X_test_MIC.shape)

# SMOTE采样
X_train_MIC_SMOTE, y_train_MIC_SMOTE =SMOTE(random_state=random_seed).fit_resample(X_train_MIC, y_train)
print('MIC_SMOTE_train_set:', Counter(y_train_MIC_SMOTE), '\n', 'test_set:', Counter(y_test))

# BorderlineSMOTE采样
X_train_MIC_B_SMOTE, y_train_MIC_B_SMOTE =BorderlineSMOTE(random_state=random_seed).fit_resample(X_train_MIC, y_train)
print('MIC_B_SMOTE_train_set:', Counter(y_train_MIC_B_SMOTE), '\n', 'test_set:', Counter(y_test))

# SMOTE+Tomek_Link采样
X_train_MIC_S_T_L, y_train_MIC_S_T_L =SMOTETomek(random_state=random_seed).fit_resample(X_train_MIC, y_train)
print('MIC_S_T_L_train_set:', Counter(y_train_MIC_S_T_L), '\n', 'test_set:', Counter(y_test))

###### 递归特征消除法
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
rfe_model = RFE(AdaBoostClassifier(random_state=random_seed))
rfe = rfe_model.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)# 最优特征
# # RFE特征选择后的 训练集 和 测试集
X_train_RFE= X_train[rfe.get_feature_names_out()]
X_test_RFE=X_test[rfe.get_feature_names_out()]
print('RFE特征选择后的训练集和特征集的维度', X_train_RFE.head(5), X_test_RFE.shape)

feature_ranking = rfe.ranking_
print(feature_ranking)
feature_importance_values = rfe.estimator_.feature_importances_
print(feature_importance_values)

FI_RFE=pd.DataFrame(feature_importance_values, index=X_train_RFE.columns, columns=['features importance'])
print(FI_RFE)

## 由高到低进行排序
FI_RFE=FI_RFE.sort_values("features importance",ascending=False).round(3)
print(FI_RFE)

# 获取重要程度大于0的系数指标
plt.figure(figsize=(15, 10))
FI_RFE[FI_RFE["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='firebrick',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=30)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.5(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()




# print(rfe.n_features_)#最优特征数
# print(rfe.support_)#打印最优特征变量
# print(rfe.ranking_)#特征排序
# print(rfe.feature_importances_)


# SMOTE采样
X_train_RFE_SMOTE, y_train_RFE_SMOTE =SMOTE(random_state=random_seed).fit_resample(X_train_RFE, y_train)
print('RFE_SMOTE_train_set:', Counter(y_train_RFE_SMOTE), '\n', 'test_set:', Counter(y_test))

# BorderlineSMOTE采样
X_train_RFE_B_SMOTE, y_train_RFE_B_SMOTE =BorderlineSMOTE(random_state=random_seed).fit_resample(X_train_RFE, y_train)
print('RFE_B_SMOTE_train_set:', Counter(y_train_RFE_B_SMOTE), '\n', 'test_set:', Counter(y_test))

# SMOTE+Tomek_Link采样
X_train_RFE_S_T_L, y_train_RFE_S_T_L =SMOTETomek(random_state=random_seed).fit_resample(X_train_RFE, y_train)
print('RFE_S_T_L_train_set:', Counter(y_train_RFE_S_T_L), '\n', 'test_set:', Counter(y_test))
#
# 经过比较后，RFE + SMOTE_Tomek_link 的预处理组合性能最好，因此将训练集和测试集保存
data_train=pd.concat([X_train_RFE_S_T_L, y_train_RFE_S_T_L], axis=1)
data_train=data_train.reset_index(drop=True)
data_test=pd.concat([X_test_RFE, y_test], axis=1)
data_test=data_test.reset_index(drop=True)
print(data_train.shape, data_test.shape)

data_train.to_csv(path_or_buf=r'data/data_train.csv', index=None)
data_test.to_csv(path_or_buf=r'data/data_test.csv', index=None)
#
# 训练模型，比较性能
def model(X_train, y_train, X_test, y_test, estimator):
    # 归一化
    mm=MinMaxScaler()
    # 训练集归一化
    X_train_std=pd.DataFrame(mm.fit_transform(X_train))
    X_train_std.columns=X_train.columns
    print('标准化后的训练集维度', X_train_std.shape)
    # 测试集归一化
    X_test_std=pd.DataFrame(mm.fit_transform(X_test))
    X_test_std.columns=X_test.columns
    print('标准化后的测试集维度', X_test_std.shape)
    # 训练模型
    est=estimator
    est.fit(X_train_std, y_train)
    y_pred=est.predict(X_test_std)
    y_pred_prob=est.predict_proba(X_test_std)
    # 输出性能
    acc=accuracy_score(y_test, y_pred)
    pre=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1_=f1_score(y_test, y_pred)
    auc=roc_auc_score(y_test, y_pred)
    print('accuracy: %f,precision: %f,recall: %f,f1_score: %f,auc: %f'% (acc, pre, rec, f1_, auc))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

XG=XGBClassifier(random_state=random_seed)
RF=RandomForestClassifier(random_state=random_seed)
LGBM=LGBMClassifier(random_state=random_seed)
LR=LogisticRegression(random_state=random_seed)

# 比较Lasso特征选择后的 三种采样方法 的性能
model(X_train_lasso_SMOTE, y_train_lasso_SMOTE, X_test_lasso, y_test, estimator=RF)
print('Above is Lasso + SMOTE')
model(X_train_lasso_B_SMOTE, y_train_lasso_B_SMOTE, X_test_lasso, y_test, estimator=RF)
print('Above is Lasso + B_SMOTE')
model(X_train_lasso_S_T_L, y_train_lasso_S_T_L, X_test_lasso, y_test, estimator=RF)
print('Above is Lasso + S_T_L')

# 比较MIC特征选择后的 三种采样方法 的性能
model(X_train_MIC_SMOTE, y_train_MIC_SMOTE, X_test_MIC, y_test, estimator=RF)
print('Above is MIC + SMOTE')
model(X_train_MIC_B_SMOTE, y_train_MIC_B_SMOTE, X_test_MIC, y_test, estimator=RF)
print('Above is MIC + B_SMOTE')
model(X_train_MIC_S_T_L, y_train_MIC_S_T_L, X_test_MIC, y_test, estimator=RF)
print('Above is MIC + S_T_L')

# 比较RFE特征选择后的 三种采样方法 的性能
model(X_train_RFE_SMOTE, y_train_RFE_SMOTE, X_test_RFE, y_test, estimator=RF)
print('Above is RFE + SMOTE')
model(X_train_RFE_B_SMOTE, y_train_RFE_B_SMOTE, X_test_RFE, y_test, estimator=RF)
print('Above is RFE + B_SMOTE')
model(X_train_RFE_S_T_L, y_train_RFE_S_T_L, X_test_RFE, y_test, estimator=RF)
print('Above is RFE + S_T_L')
#
# 查看数据分布
# 首先看看平衡前的原始分布
# kernelPCA 2D projection
# 归一化
mm=MinMaxScaler()
X_train_std=pd.DataFrame(mm.fit_transform(X_train))
X_train_std.columns=X_train.columns
print('标准化后的训练集维度', X_train_std.shape)

plot_2Dprojection_and_cardinality(X_train_std, y_train)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.xticks([0, 1], ['Not fraud', 'Fraud'], rotation='horizontal')
plt.legend(loc='upper right')
plt.show()

# 再看看平衡后的数据分布
X_train_RFE_S_T_L_std=pd.DataFrame(mm.fit_transform(X_train_RFE_S_T_L))
X_train_RFE_S_T_L_std.columns=X_train_RFE_S_T_L.columns
plot_2Dprojection_and_cardinality(X_train_RFE_S_T_L_std, y_train_RFE_S_T_L)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.legend(loc='upper right')
plt.xticks([0, 1], ['Not fraud', 'Fraud'], rotation='horizontal')
plt.show()