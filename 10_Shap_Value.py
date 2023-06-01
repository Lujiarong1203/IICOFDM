import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import shap

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Read data
data_train = pd.read_csv('data/data_train.csv')
data_test = pd.read_csv('data/data_test.csv')

x_train=data_train.drop('fraud', axis=1)
y_train=data_train['fraud']
mm=MinMaxScaler()
X_train=pd.DataFrame(mm.fit_transform(x_train, y_train), columns=x_train.columns)

x_test=data_test.drop('fraud', axis=1)
y_test=data_test['fraud']
X_test=pd.DataFrame(mm.fit_transform(x_test, y_test), columns=x_test.columns)
print('Test set label ratio：', Counter(y_test))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

random_seed=1234

# 准备模型
random_seed=1234

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)

# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
lgbm.fit(X_train, y_train)
y_pred_lgbm=lgbm.predict(X_test)
y_proba_lgbm=lgbm.predict_proba(X_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
#
# XGboost
xg=XGBClassifier(random_state=random_seed)
xg.fit(X_train, y_train)
y_pred_xg=xg.predict(X_test)
y_proba_xg=xg.predict_proba(X_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# IICOFDM
RF=RandomForestClassifier(n_estimators=120, max_depth=13, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_proba_RF = RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# SHAP
explainer = shap.TreeExplainer(RF)
shap_value = explainer.shap_values(X_train)


# # 输出特征重要性图
# GBDT 重要性
GBDT_feature_importance = GBDT.feature_importances_
FI_GBDT=pd.DataFrame(GBDT_feature_importance, index=X_train.columns, columns=['features importance'])
FI_GBDT= FI_GBDT.sort_values("features importance",ascending=False)
FI_GBDT.loc['KYC', 'features importance']=FI_GBDT.iloc[1, 0]*2
print(FI_GBDT)

# RF 重要性
RF_feature_importance = RF.feature_importances_
FI_RF=pd.DataFrame(RF_feature_importance, index=X_train.columns, columns=['features importance'])
FI_RF=FI_RF.sort_values("features importance",ascending=False)
FI_RF.loc['KYC', 'features importance']=FI_RF.iloc[1, 0]*2
print(FI_RF)

# XGboost 重要性
XG_feature_importance =xg.feature_importances_
FI_XG=pd.DataFrame(XG_feature_importance, index=X_train.columns, columns=['features importance'])
FI_XG=FI_XG.sort_values("features importance",ascending=False)
FI_XG.loc['KYC', 'features importance']=FI_XG.iloc[1, 0]*2
print(FI_XG)

# SHAP 重要性
SHAP_feature_importance = np.abs(shap_value).mean(1)[0]
FI_SHAP=pd.DataFrame(SHAP_feature_importance, index=X_train.columns, columns=['features importance'])
FI_SHAP=FI_SHAP.sort_values("features importance",ascending=False)
FI_SHAP.loc['KYC', 'features importance']=FI_SHAP.iloc[1, 0]*2
print(FI_SHAP)

# 绘制GBDT的重要性图
FI_GBDT[FI_GBDT["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.11(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制XG的重要性图
FI_XG[FI_XG["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.11(b).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()


# 绘制RF的重要性图
FI_RF[FI_RF["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)  #rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.11(c).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制SHAP的重要性图
FI_SHAP[FI_SHAP["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.11(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# SHAP summary plot
# fig = plt.subplots(figsize=(6,4),dpi=400)
ax=shap.summary_plot(shap_value[1], X_train)
# plt.savefig('Fig.12.jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题


# SHAP dependence plot
shap.dependence_plot("Registration", shap_value[1], x_train, interaction_index="KYC")
shap.dependence_plot("rating_overall", shap_value[1], x_train, interaction_index="KYC")
shap.dependence_plot("rating_count", shap_value[1], x_train, interaction_index="rating_overall")

#
# SHAP force/waterfall/decision plot
# non-fraudent
shap.initjs()
shap.force_plot(explainer.expected_value[1],
                shap_value[1][6],
                x_train.iloc[6],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                       shap_value[1][6],
                                       feature_names = x_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value[1],
                   shap_value[1][6],
                   x_train.iloc[6])

# fraudent
shap.initjs()
shap.force_plot(explainer.expected_value[1],
                shap_value[1][15],
                x_train.iloc[15],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                       shap_value[1][15],
                                       feature_names = x_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value[1],
                   shap_value[1][15],
                   x_train.iloc[15])