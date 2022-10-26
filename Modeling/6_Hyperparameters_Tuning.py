import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Load the data
data = pd.read_csv('C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_train.csv')
print(data.shape)

y_train=data['fraud']
x_train=data.drop('fraud', axis=1)
print(x_train.shape, y_train.shape)

# stratified K-fold
stra_kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
cnt=1
for train_index, test_index in stra_kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# Parameter tuning/Each time a parameter is tuned, update the parameter corresponding to other_params to the optimal value
# Use more trees (n_estimators)
cv_params= {'n_estimators': [15, 20, 25, 30, 35, 40]}

model = RandomForestClassifier(random_state=1234)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=stra_kf, verbose=1, n_jobs=-1)
optimized_RF.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the n_estimators validation_curve
param_range_1=[15, 20, 25, 30, 35, 40]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=stra_kf, scoring='f1', n_jobs=-1)

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

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('f1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) n_estimators', y=-0.25, fontsize=15)
plt.ylim([0.9, 1.0])
plt.tight_layout()
plt.show()
#
# tuning max_depth
cv_params= {'max_depth': [2, 3, 4, 5, 6, 7, 8]}

model = RandomForestClassifier(n_estimators=25, random_state=1234)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=stra_kf, verbose=1, n_jobs=-1)
optimized_RF.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the max_depth validation curve
param_range_2=[2, 3, 4, 5, 6, 7, 8]
train_scores_2, test_scores_2 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='max_depth',
                                             param_range=param_range_2,
                                             cv=stra_kf, scoring='f1', n_jobs=-1)

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

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('f1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) max_depth', y=-0.25, fontsize=15)
plt.ylim([0.85, 1.0])
plt.tight_layout()
plt.show()
#
# tuning max_features
cv_params= {'max_features': [2, 3, 4, 5, 6, 7, 8]}

model = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=1234)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=stra_kf, verbose=1, n_jobs=-1)
optimized_RF.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# Draw the max_features validation curve
param_range_3=[2, 3, 4, 5, 6, 7, 8]
train_scores_3, test_scores_3 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='max_features',
                                             param_range=param_range_3,
                                             cv=stra_kf, scoring='f1', n_jobs=-1)

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

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('f1-Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) max_features', y=-0.25, fontsize=15)
plt.ylim([0.75, 1.0])
plt.tight_layout()
plt.show()
#
# tuning min_samples_leaf
cv_params= {'min_samples_leaf': [1, 2, 3, 4]}
model = RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4, random_state=1234)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=stra_kf, verbose=1, n_jobs=-1)
optimized_RF.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score:{0}'.format(optimized_RF.best_score_))

# 调试min_samples_split
cv_params= {'min_samples_split': [2, 3, 4, 5, 6]}

model = RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4, min_samples_leaf=1, random_state=1234)
optimized_RF = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=stra_kf, verbose=1, n_jobs=-1)
optimized_RF.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_RF.best_params_))
print('Best model score分:{0}'.format(optimized_RF.best_score_))
#
# #The optimal parameters：(n_estimators=25, max_depth=4, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=1234)
#
data_test=pd.read_csv('data/data_test.csv')
print(data_test.shape)
#
# Verify that the optimal parameters improve the effect
x_test=data_test.drop(['fraud'], axis=1)
y_test=data_test['fraud']
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


score_data_0=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(RandomForestClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data_0 = score_data_0.append(pd.DataFrame({'RF Before tuning': [score.mean()]}), ignore_index=True)
# print(score_data_0)

score_data_1=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data_1 = score_data_1.append(pd.DataFrame({'RF After tuning': [score.mean()]}), ignore_index=True)
# print(score_data_1)

score_com=pd.concat([score_data_0, score_data_1], axis=1)
score_COM=score_com.rename(index={0:'accuracy', 1:'precision', 2:'recall', 3:'f1', 4:'roc_auc'})

print(score_COM.T)
