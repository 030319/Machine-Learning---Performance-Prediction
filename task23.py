import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
#文件路径:
filepath0=r'C:\Users\86135\Desktop\工作\计算机一级项目\第二三阶段\学生成绩记录\文本挖掘与推荐系统.csv'
filepath1=r'C:\Users\86135\Desktop\工作\计算机一级项目\第二三阶段\学生成绩记录\计算机硬件基础.csv'
data0 = pd.read_csv(filepath0,encoding='gb2312')
data0['是否及格']=np.where(data0.总分>=60,1,0)
data1 = pd.read_csv(filepath1,encoding='gb2312')
data1['是否及格']=np.where(data1.总分>=60,1,0)
# for row in data.itertuples():
#     print(getattr(row, '总分'), getattr(row, '是否及格')) # 输出每一行
#读取前五行数据：
# data=pd.read_csv(filepath,header=None,encoding="gbk",names=["平时成绩","期末成绩","总分"],skiprows=1,skipfooter=1,usecols=["平时成绩","期末成绩","总分"],index_col=0,engine="python")
print(data0.head())
print(data1.head())
print('数据集大小',data0.shape)
print('数据集大小',data1.shape)
#寻找学生数据中是否存在空值
#判断特征类型
print(data0.isnull().any())
print(data0.dtypes)
print(data1.isnull().any()) 
print(data1.dtypes)
data0.describe() #仅对数值型数据进行统计
print(data0.describe(include='all'))#对所有的数据进行统计分析
data1.describe()
print(data1.describe(include='all'))
#各特征系数与总评的相关系数,降序排列
corrDf0=data0.corr()
print(corrDf0['总分'].sort_values(ascending =False))
corrDf1=data1.corr()
print(corrDf1['总分'].sort_values(ascending =False))
x0=data0.loc[:,['平时成绩','期末成绩']]
y0=data0.loc[:,['是否及格']]
x0_train,x0_test, y0_train, y0_test =sklearn.model_selection.train_test_split(x0,y0,test_size=0.4, random_state=5)
# print(x0_train.shape, x0_test.shape, y0_train.shape, y0_test.shape)
clf0 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf0 = clf0.fit(x0_train, y0_train.astype('int'))
y0_predict = clf0.predict(x0_test)
# display(y0_predict.shape,y0_test.shape)
# display(y0_predict,y0_test)
y0_test=y0_test.astype(np.int64)
print('预测是否及格的准确率为{:.2f}%'.format(accuracy_score(y0_test,y0_predict)*100))
print(' ')
x1=data1.loc[:,['作业成绩','实践成绩','平时成绩','期末成绩']]
y1=data1.loc[:,['是否及格']]
x1_train,x1_test, y1_train, y1_test =sklearn.model_selection.train_test_split(x1,y1,test_size=0.4, random_state=5)
# print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape)
clf1 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf1 = clf1.fit(x1_train, y1_train.astype('int'))
y1_predict = clf1.predict(x1_test)
# display(y1_predict.shape,y1_test.shape)
# display(x1_text,y1_test)
y1_test=y1_test.astype(np.int64)
print('预测是否及格的准确率为{:.2f}%'.format(accuracy_score(y1_test,y1_predict)*100))
print(' ')
x0=data0.loc[:,['平时成绩','期末成绩']]
y0=data0.loc[:,['总分']]
x0_train,x0_test, y0_train, y0_test =sklearn.model_selection.train_test_split(x0,y0,test_size=0.4, random_state=5)
clf0 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf0 = clf0.fit(x0_train, y0_train.astype('int'))
y0_predict = clf0.predict(x0_test)
y0_test=y0_test.astype(np.int64)
print(x0_test,y0_test)
print('预测准确分数的准确率为{:.2f}%'.format(accuracy_score(y0_test,y0_predict)*100))
print('预测分数为',y0_predict)
print(' ')
x1=data1.loc[:,['作业成绩','实践成绩','平时成绩','期末成绩']]
y1=data1.loc[:,['总分']]
x1_train,x1_test, y1_train, y1_test =sklearn.model_selection.train_test_split(x1,y1,test_size=0.4, random_state=5)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
clf1 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf1 = clf1.fit(x1_train, y1_train.astype('int'))
y1_predict = clf1.predict(x1_test)
# display(y1_predict.shape,y1_test.shape)
print(x1_test,y1_test)
y1_test=y1_test.astype(np.int64)
print('决策树预测分数的准确率为{:.2f}%'.format(accuracy_score(y1_test,y1_predict)*100))
print('预测准确分数的准确率为',y1_predict)
print(' ')
rf=RandomForestClassifier(n_estimators = 100, oob_score = True, n_jobs = -1,random_state =42,max_features = 'auto', min_samples_leaf = 12)
rf.fit(x1_train, y1_train.astype('int'))
y_predict0 = rf.predict(x1_test)
# display(y_test,y_predict)
print('随机森林预测分数的准确率为{:.2f}%'.format(metrics.accuracy_score(y1_test, y_predict0)*100))
error0 = mean_absolute_error(y1_test,y_predict0)
print('当使用随机森林默认参数时，平均绝对误差为：', error0)
print(y_predict0)
print(' ')
# print(grid_search_rf.best_params_)
rf1 = RandomForestRegressor(bootstrap=True,
                             max_depth=10,
                             max_features='auto',
                             min_samples_leaf=1,
                             min_samples_split=2,
                             n_estimators=20,
                            )
rf1.fit(x1_train, y1_train)   #对照着y_train作为目标数据训练X_train数据
y_predict1 = rf1.predict(x1_test)  #  开始预测之前就分好的测试集
error1 = mean_absolute_error(y1_test, y_predict1)
print('调参后，平均绝对误差为：', error1)
# display(y_predict0)
print('调参后预测的分数为',y_predict1)



#逻辑回归预测
from sklearn.linear_model import LogisticRegression
np.random.seed(888)
#-----训练模型--------------------
clf = LogisticRegression(random_state=0)
clf = clf.fit(x1_train, y1_train.astype('int'))
#------打印结果------------------------
print("模型参数："+str(clf.coef_))
print("模型阈值："+str(clf.intercept_))
score=clf.score(x1_test,y1_test)
print('预测准确率为{:.2f}%'.format(score*100))
pred2=clf.predict(x1_test)
print(pred2)