import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

from sklearn.model_selection import KFold

import scikitplot as skplt

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

# learning_curve
# LR
LR=LogisticRegression(random_state=random_seed)
# KNN
knn=KNeighborsClassifier()
# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
# XGboost
xg=xgboost.XGBClassifier(random_state=random_seed)
# Adaboost
Ada=AdaBoostClassifier(random_state=random_seed)
# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
# ET
ETC=ExtraTreesClassifier(random_state=random_seed)
# Random Forest
RF=RandomForestClassifier(n_estimators=120, max_depth=13, max_features=4,
                          min_samples_leaf=1, min_samples_split=2, random_state=random_seed)
#
# K-fold
kf=KFold(n_splits=5, shuffle=True, random_state=1234)
cnt=1
for train_index, test_index in kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

skplt.estimators.plot_learning_curve(LR, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(GBDT, X_train, y_train, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) GBDT', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(Ada, X_train, y_train, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) Ada', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

# #
skplt.estimators.plot_learning_curve(lgbm, X_train, y_train, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(d) LightGBM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()
#
skplt.estimators.plot_learning_curve(xg, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(e) XGboost', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(RF, X_train, y_train, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(f) IICOFDM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()