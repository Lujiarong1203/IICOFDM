import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import xgboost
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold

import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

data_train=pd.read_csv('data/data_train.csv')
x_train=data_train.drop('fraud', axis=1)
y_train=data_train['fraud']
print(x_train.shape, y_train.shape)

# learning_curve
# LR
LR=LogisticRegression(penalty='l2', C=1, max_iter=100,
                      class_weight={0:0.1,1:0.9}, random_state=1234)
LR.fit(x_train, y_train)

# KNN
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)

# LightGBM
lgbm=LGBMClassifier(n_estimators=300, max_depth=6, random_state=1234)
lgbm.fit(x_train, y_train)

# XGboost
xg=xgboost.XGBClassifier(n_estimators=300, max_depth=6, random_state=1234)
xg.fit(x_train, y_train)

# Adaboost
Ada=AdaBoostClassifier(random_state=1234)
Ada.fit(x_train, y_train)

# GBDT
GBDT=GradientBoostingClassifier(random_state=1234)
GBDT.fit(x_train, y_train)

# ET
ETC=ExtraTreesClassifier(random_state=1234)
ETC.fit(x_train, y_train)

# Random Forest
rf=RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4,
                          min_samples_leaf=1, min_samples_split=2, random_state=1234)
rf.fit(x_train, y_train)

# stratified K-fold
stra_kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
cnt=1
for train_index, test_index in stra_kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

skplt.estimators.plot_learning_curve(LR, x_train, y_train, title=None, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

skplt.estimators.plot_learning_curve(xg, x_train, y_train, title=None, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) XGboost', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()
#
skplt.estimators.plot_learning_curve(lgbm, x_train, y_train, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) LightGBM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()
#
skplt.estimators.plot_learning_curve(GBDT, x_train, y_train, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(d) GBDT', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(ETC, x_train, y_train, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(e) ET', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()
#
skplt.estimators.plot_learning_curve(rf, x_train, y_train, cv=stra_kf, random_state=1234, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(f) IICOFDM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()