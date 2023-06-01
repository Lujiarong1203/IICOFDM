import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from collections import Counter

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data= pd.read_csv("data/data_1.csv")
print(data.shape)

print(Counter(data['fraud']))

# 查看缺失值

data_miss=data[['KYC', 'Registration', 'rating_marketing', 'rating_overall', 'rating_count',
                'rating_finance', 'Bounty', 'Verified_team', 'type', 'fraud']]

print(data_miss.isnull().sum())

na_ratio=data_miss.isnull().sum()[data_miss.isnull().sum()>=0].sort_values(ascending=False)/len(data_miss)
na_sum=data_miss.isnull().sum().sort_values(ascending=False)
print(na_ratio)

missng1=msno.matrix(data_miss,labels=True,label_rotation=0,fontsize=15,figsize=(15, 10))#绘制缺失值矩阵图
plt.xticks(fontsize=25, rotation=30)
plt.yticks(fontsize=25, rotation=30)
plt.savefig('Fig.4(a).jpg', bbox_inches='tight',pad_inches=0,dpi=1500,)
plt.show()

# 查看缺失值情况 /  绘制缺失比例图
def null_hist(data_missing):         # 自定义缺失值比例绘图函数
    cols = list(data_missing.columns)
    the_null_percents = []
    for col in cols:
        the_null_percent = np.uint8(data_missing[col].isnull()).sum()/len(data_missing[col].values)   # 计算特征列的缺失值比例
        the_null_percent = round(the_null_percent,2)#保留两位小数
        the_null_percents.append(the_null_percent)

    plot_data = pd.Series(the_null_percents, index = cols)    # 将比例值和对应的特征名称构建成一个Series
#    plot_data = plot_data[plot_data.values>0.01]              # 只取出比例大于0.001的数据绘制直方图
    plot_data = plot_data.sort_values(ascending=False)        # 对数据进行降序排列
    plot_data = plot_data[plot_data.values != 0]              # 取出其中比例值不等于o的数据作为最终用于绘图的数据

    ## 绘制直方图
    plot_x = plot_data.index
    plot_y = plot_data.values

    plt.figure(dpi=150,figsize=(15,10))#dpi分辨率
    plt.bar(plot_x, plot_y, edgecolor='none', color='cornflowerblue',alpha=0.8,width=0.8)

    plt.rcParams['axes.unicode_minus'] = False       # 图中可以显示负号

    plt.xlabel("Feature name",fontsize=25)
    plt.ylabel("Missing proportion",fontsize=25)

    plt.tick_params(labelsize = 25)         #设置坐标轴数字的大小
    # plt.xticks(rotation = 30)               #设置坐标轴轴上注记字发生旋转

    for a,b in zip(plot_x,plot_y.round(3)):                    #在柱子上注释数字
        plt.text(a,b+0.01,b,ha='center',va='bottom',fontsize=15)
        #a指示x的位置，b+50指示y的位置，第二个b为条柱上的注记数字,ha表示注记数字的对齐方式，fontsize表示注释数字字体大小
        #va表示条柱位于注释数字底部还是顶部
    plt.savefig('Fig.4(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)  # 解决图片不清晰，不完整的问题
    plt.show()
null_hist(data_miss)    # 调用函数绘制缺失值直方图# 查看缺失值情况 /  绘制缺失比例图

# # Fill rating_count with the mode
data['rating_count'].fillna(float(data['rating_count'].mode()), inplace=True)

# Fill in the following features with the average
data['rating_finance']=data['rating_finance'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_finance']=data['rating_finance'].fillna(data['rating_finance'].astype(float).mean())

data['rating_marketing']=data['rating_marketing'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_marketing']=data['rating_marketing'].fillna(data['rating_marketing'].astype(float).mean())
#
data['rating_overall']=data['rating_overall'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_overall']=data['rating_overall'].fillna(data['rating_overall'].astype(float).mean())

# Fill in the missing value of type with the previous line
# data['type'].fillna(method='pad', axis=0, inplace=True)
index=data.type
print(index.value_counts())
print(data['type'].mode())
data['type']=data['type'].fillna('Utility-token')

print('After filling in the missing values：', '\n', data.isnull().sum())
print('The type of each features', data.dtypes)

# Converts partial feature types
columns=['rating_finance', 'rating_marketing', 'rating_overall']
for k in columns:
    data[k]=data[k].apply(lambda x: float(x))
print('The converted data type：', '\n', data.dtypes)
#
data.to_csv(path_or_buf=r'data/data_2.csv', index=None)