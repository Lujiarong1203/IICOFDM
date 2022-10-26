import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from math import sqrt
import shap
from shap.plots import _waterfall
from IPython.display import (display, display_html, display_png, display_svg)

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# read data
data_train = pd.read_csv('C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_train.csv')
data_test = pd.read_csv('C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_test.csv')
print(data_train.shape, data_test.shape)


x_train=data_train.drop('fraud', axis=1)
y_train=data_train['fraud']

x_test=data_test.drop('fraud', axis=1)
y_test=data_test['fraud']

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# rf
RF=RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=1234)
RF.fit(x_train, y_train)
y_pred_RF = RF.predict(x_test)
y_proba_RF = RF.predict_proba(x_test)

print(x_train.shape)
explainer = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(x_train)
shap_values_2=explainer(x_train)
shap_value=np.array(shap_values[1])
print(shap_value[0, :])

# summary_plot
# shap.summary_plot(shap_value, x_train, max_display=18)

# dependent plot特征依赖分析图
shap.dependence_plot('Registration', shap_value, x_train, interaction_index='KYC')
shap.dependence_plot('rating_count', shap_value, x_train, interaction_index='KYC')
shap.dependence_plot('rating_overall', shap_value, x_train, interaction_index='Registration')

# shap.dependence_plot('Premium', shap_value, x_train, interaction_index='KYC')

#shap force plot
# shap.initjs()
# shap.force_plot(explainer.expected_value[1],
#                 shap_value[3,:],
#                 x_train.iloc[3,:],
#                 text_rotation=20,
#                 matplotlib=True)
#
# shap.force_plot(explainer.expected_value[1],
#                 shap_value[7,:],
#                 x_train.iloc[7,:],
#                 text_rotation=20,
#                 matplotlib=True)

global_shap_values = pd.DataFrame(np.abs(shap_value).mean(0),index=x_train.columns).reset_index()
global_shap_values.columns = ['feature','scores']
global_shap_values = global_shap_values.sort_values('scores',ascending=False)
shap_FI=global_shap_values['feature'][:8]
y_pos_shap=np.arange(len(shap_FI))
scores_shap = global_shap_values['scores'][:10]
FI_shap = pd.DataFrame(list(zip(shap_FI, scores_shap)))
FI_shap.columns = ['features','scores']

print(FI_shap)

plt.figure(figsize=(20,30),dpi=150)
ax1 = plt.subplot(111)

# plt.sca(ax1)
plt.barh(y=FI_shap.loc[:,'features'],
         width=FI_shap.sort_values('scores',ascending=True).loc[:,'scores'],color='crimson',alpha=0.8)
plt.yticks(y_pos_shap,FI_shap.sort_values('scores',ascending=True).loc[:,'features'],fontsize=30)
plt.legend(['Feature importances(SHAP)'])

plt.show()