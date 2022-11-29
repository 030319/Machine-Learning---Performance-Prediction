from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 把数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
"""
(455, 30) (455,)
(114, 30) (114,)
"""
"""

# 使用LogisticRegression模型来训练，并计算训练集的评分数据和测试集的评分数据。
# 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('train score: {train_score: .6f}; test score: {test_score: .6f}'.format(train_score=train_score, test_score=test_score))
"""
train score:  0.958242; test score:  0.947368
"""


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 增加多项式预处理
def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("logistic_regression", logistic_regression)])
    return pipeline

# 接着，增加二项多项式特征，创建并训练模型
import time
model = polynomial_model(degree=2, penalty='l1') #我们使用L1范数作为正则项（参数penalty='l1'）

start = time.clock()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1: 0.6f}; cv_score: {2: .6f}'.format(time.clock()-start, train_score, cv_score))

"""输出：
elaspe: 0.369242; train_score:  0.993407; cv_score:  0.964912
"""
