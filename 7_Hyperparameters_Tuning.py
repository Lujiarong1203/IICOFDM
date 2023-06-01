import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data
data_train = pd.read_csv('data/data_train.csv')
print(data_train.shape)

y_train=data_train['fraud']
x_train=data_train.drop('fraud', axis=1)
print(x_train.shape, y_train.shape)
print(x_train.head(5))

# 归一化
mm=MinMaxScaler()
X_train=pd.DataFrame(mm.fit_transform(x_train, y_train), columns=x_train.columns)
print(X_train.head(5))

random_seed=1234

# K-fold
kf=KFold(n_splits=7, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in kf.split(X_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# Parameter tuning/Each time a parameter is tuned, update the parameter corresponding to other_params to the optimal value
# Use more trees (n_estimators)
cv_params= {'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]}

model = RandomForestClassifier(random_state=random_seed)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=kf, verbose=1, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the n_estimators validation_curve
param_range_1=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=kf, scoring='f1', n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="darkorange")

plt.plot(param_range_1, test_mean_1, color="blue", linewidth=3.0,
         marker='d', markersize=10,label='validation score')

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="blue")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('F1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) n_estimators', y=-0.25, fontsize=15)
plt.ylim([0.9825, 1.0])
plt.tight_layout()
plt.show()




# tuning max_depth
cv_params= {'max_depth': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}

model = RandomForestClassifier(n_estimators=120, random_state=random_seed)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=kf, verbose=1, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the max_depth validation curve
param_range_2=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
train_scores_2, test_scores_2 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='max_depth',
                                             param_range=param_range_2,
                                             cv=kf, scoring='f1', n_jobs=-1)

train_mean_2=np.mean(train_scores_2, axis=1)
train_std_2=np.std(train_scores_2, axis=1)
test_mean_2=np.mean(test_scores_2, axis=1)
test_std_2=np.std(test_scores_2, axis=1)

plt.plot(param_range_2, train_mean_2, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_2, train_mean_2 + train_std_2,
                 train_mean_2 - train_std_2, alpha=0.1, color="darkorange")

plt.plot(param_range_2, test_mean_2, color="blue", linewidth=3.0,
         marker='d', markersize=10, label='validation score')

plt.fill_between(param_range_2, test_mean_2 + test_std_2,
                 test_mean_2 - test_std_2, alpha=0.1, color="blue")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('F1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) max_depth', y=-0.25, fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()





# tuning max_features
cv_params= {'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

model = RandomForestClassifier(n_estimators=120, max_depth=13, random_state=random_seed)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=kf, verbose=1, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the max_features validation curve
param_range_3=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores_3, test_scores_3 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='max_features',
                                             param_range=param_range_3,
                                             cv=kf, scoring='f1', n_jobs=-1)

train_mean_3=np.mean(train_scores_3, axis=1)
train_std_3=np.std(train_scores_3, axis=1)
test_mean_3=np.mean(test_scores_3, axis=1)
test_std_3=np.std(test_scores_3, axis=1)

plt.plot(param_range_3, train_mean_3, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_3, train_mean_3 + train_std_3,
                 train_mean_3 - train_std_3, alpha=0.1, color="darkorange")

plt.plot(param_range_3, test_mean_3, color="blue", linewidth=3.0,
         marker='d', markersize=10, label='validation score')

plt.fill_between(param_range_3, test_mean_3 + test_std_3,
                 test_mean_3 - test_std_3, alpha=0.1, color="blue")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('F1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) max_features', y=-0.25, fontsize=15)
plt.ylim([0.982, 1.0])
plt.tight_layout()
plt.show()





# tuning min_samples_leaf
cv_params= {'min_samples_leaf': [1, 2, 3, 4]}
model = RandomForestClassifier(n_estimators=120, max_depth=13, max_features=4, random_state=random_seed)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=kf, verbose=1, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))
#
#
#
# 调试min_samples_split
cv_params= {'min_samples_split': [2, 3, 4, 5, 6]}

model = RandomForestClassifier(n_estimators=120, max_depth=13, max_features=4, min_samples_leaf=1, random_state=random_seed)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=kf, verbose=1, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score分:{0}'.format(optimized_RF.best_score_))
# #
# #
# #
# #
# #
# # #The optimal parameters：(n_estimators=120, max_depth=13, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=1234)
# #
data_test=pd.read_csv('data/data_test.csv')
print(data_test.shape)
# #
# Verify that the optimal parameters improve the effect
x_test=data_test.drop(['fraud'], axis=1)
y_test=data_test['fraud']
X_test=pd.DataFrame(mm.fit_transform(x_test, y_test), columns=x_test.columns)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 比较超参数调优前后模型性能
def model_comparison(X_train, y_train, X_test, y_test, estimator):

    # # estimator
    est = estimator
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    score_data = []
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)

# 准备模型
# 调优前
RF=RandomForestClassifier(random_state=random_seed)
# 调优后
RF_Tuning=RandomForestClassifier(n_estimators=120,
                                 max_depth=13,
                                 max_features=4,
                                 min_samples_leaf=1,
                                 min_samples_split=2,
                                 random_state=random_seed
                                 )

model_comparison(X_train, y_train, X_test, y_test, RF)
model_comparison(X_train, y_train, X_test, y_test, RF_Tuning)