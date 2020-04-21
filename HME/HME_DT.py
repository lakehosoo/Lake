import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_value = pd.read_excel('HME_Data_V2.xlsx', sheet_name='Rearrange', index_col=0, header=None, skiprows=2).transpose()
data_index = pd.read_excel('HME_Data_V2.xlsx', sheet_name='Rearrange', index_col=0, header=None,nrows=2, dtype=object).transpose()
#data_value = pd.read_csv('HME/data.csv', index_col=0, header=None, skiprows=2).transpose()
#data_index = pd.read_csv('HME/data.csv', index_col=0, header=None,nrows=2, dtype=object).transpose()
data_index.columns = ["Name","Grade"]

# Grade Information
for i in range(data_index.shape[0]):
    if data_index.iloc[i,1]==0:
        data_index.iloc[i,1] = data_index.iloc[i-1,1]

data = pd.concat([data_index,data_value],axis=1)
#data.describe()
#data.info()

# Ref Average
data['Grade'] = data['Grade'].fillna(0) #Replace NA with a scalar value 0
data_ref = data[data['Grade'].str.contains('REF')==1] #data의 Grade 열 중 REF를 포함한 값만 추출
series_ref_mean = data_ref.groupby('Grade').Prop1.mean() #data_ref 중 Grade 그룹 별 Prop1 평균을 Series type 데이터로 저장

# Simple Average of Each Properties 
for Grade in series_ref_mean.index: #series_ref_mean의 index Grade 수 만큼 반복 [Ref1, Ref2,...]
    for i in range(data.shape[0]): #data 열 개수 만큼 i 반복 [i=1, 2, 3...]
        if Grade==data.iloc[i,1]: 
            data.loc[i+1,'Prop1'] = data.loc[i+1,'Prop1']/series_ref_mean[Grade]

data_ref = data[data['Grade'].str.contains('REF')==1]
data_arr = data_ref.to_numpy()
X_tot = data_arr[:,2:44]
y_tot = data_arr[:,45]

#####################################################
# Data Split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_tot, y_tot, random_state=1)

#####################################################
# Decision Tree
from sklearn import tree
from sklearn import svm

model = tree.DecisionTreeRegressor(random_state=0, max_depth=None)
model.fit(X_train, y_train)
print("Depth of Tree : {}".format(model.get_depth()))
print("Score of Train / Test : {:.3f} / {:.3f}".format(model.score(X_train, y_train),model.score(X_test, y_test)))

# Graphviz
import graphviz
dot_graph = tree.export_graphviz(model)
dot = graphviz.Source(dot_graph,format='pdf')
dot.render('tree')

def plot_feature_importances(model):
    n_features = X_tot.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_value.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature Name")
    plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()

i= model.get_depth()
depth = []
s_train = []
s_test = []
while i > 4 :
    model = tree.DecisionTreeRegressor(random_state=0, max_depth=i)
    model.fit(X_train, y_train)
    depth.append(model.get_depth())
    s_train.append(model.score(X_train, y_train))
    s_test.append(model.score(X_test, y_test))
    i = model.get_depth() - 1
plt.plot(depth,s_train,depth,s_test)
plt.xlabel('Tree Depth')
plt.ylabel('R2 Score')
plt.legend(['Train','Test'])
plt.show()

#####################################################
# Random Forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
print("Score of Train / Test : {:.3f} / {:.3f}".format(model.score(X_train, y_train),model.score(X_test, y_test)))

plot_feature_importances(model)
plt.show()
'''
트리가 많을수록 random_state 값의 변화에 따른 변동이 적음
차원이 높고 희소한 데이터에는 잘 작동하지 않음. 선형 모델이 더 적합함
n_estimators는 클수록 좋음 : 많은 트리를 평균하면 과대적합을 줄여 더 안정적인 모델
max_features는 각 트리가 얼마나 무작위가 될지를 결정하며 작은 값은 과대적합을 줄임
'''
#####################################################
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=0) # default (Max_depth: 3, n_estimators: 100, Learning Rate: 0.1)
model.fit(X_train, y_train)
print("Score of Train / Test : {:.3f} / {:.3f}".format(model.score(X_train, y_train),model.score(X_test, y_test)))

model = GradientBoostingRegressor(random_state=0, max_depth=5, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
print("Score of Train / Test : {:.3f} / {:.3f}".format(model.score(X_train, y_train),model.score(X_test, y_test)))

plot_feature_importances(model)
plt.show()
'''
learning_rate : 이전 트리의 오차를 얼마나 강하게 보정할 것인지를 제어
learning_rate를 낮추면 비슷한 복잡도의 모델을 만들기 위해서 더 많은 트리를 추가해야 함
Randon Forest와는 달리 n_estimators를 크게 하면 모델이 복잡해지고 과대적합될 가능성
가용한 시간과 메모리 한도에서 n_estimators를 맞추고 나서 적절한 learning_rate를 찾아야 함
'''
# PSO
from pyswarm import pso

score_train = []
score_test = []
x1 = []
x2 = []
x3 = []

def tuning(x):
    x1.append(np.int(x[0]))
    x2.append(np.int(x[1]))
    x3.append(x[2])
    model = GradientBoostingRegressor(random_state=0, 
                    max_depth=np.int(x[0]), n_estimators=np.int(x[1]), learning_rate=x[2])
    model.fit(X_train, y_train)
    score1 = model.score(X_train, y_train)
    score2 = model.score(X_test, y_test)
    score_train.append(score1)
    score_test.append(score2)
    print(f"Function Call : {len(x1)}, Objective : {-(score1 + score2)}")
    return -(score1 + score2)

lb = [1,  20, 0.01]
ub = [9, 100, 0.30]

xopt, fopt = pso(tuning, lb, ub, ieqcons=[], f_ieqcons=None, maxiter=1000, minstep=1e-8, minfunc=-200)
      
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(score_train,score_test,s=3,alpha=0.2)
ax.set_xlabel('score_train')
ax.set_ylabel('score_test')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.show()

model = GradientBoostingRegressor(random_state=0, 
                    max_depth=np.int(xopt[0]), n_estimators=np.int(xopt[1]), learning_rate=xopt[2])
model.fit(X_train, y_train)
print("Best Score (Train / Test) : {:.3f} / {:.3f}"
      .format(model.score(X_train, y_train),model.score(X_test, y_test)))
print("Best max_depth : {:.0f}\n"
      "Best n_estimator : {:.0f}\n"
      "Best learning_rate : {:.3f}\n"
      .format(np.int(xopt[0]), np.int(xopt[1]), xopt[2]))

#####################################################
# Gradient Boosting - Result
from sklearn.metrics import mean_squared_error

model = GradientBoostingRegressor(random_state=0, 
                    max_depth=6, n_estimators=45, learning_rate=0.14811567210358484)
model.fit(X_train, y_train)
print("Best Score (Train / Test) : {:.3f} / {:.3f}"
      .format(model.score(X_train, y_train),model.score(X_test, y_test)))

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

plt.scatter(y_train, y_pred_train, c='b', s=10, alpha=0.5)
plt.scatter(y_test, y_pred_test, c='r', s=10, alpha=0.5)
plt.legend(["Train","Test"])
plt.title("Gradient Boosting")
plt.xlabel("y")
plt.ylabel("Predicted y")
plt.axis('square')
plt.xlim(0.7,1.4)
plt.ylim(0.7,1.4)
plt.text(0.72, 1.2, f'R2 (Train / Test) = {model.score(X_train, y_train):.3f} / {model.score(X_test, y_test):.3f}' 
         , color='blue', fontsize=10)
#plt.axis([0,1.5,0,1.5])
plt.show()

#####################################################
# XGBoost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

param = {'max_depth': 6, 'eta': 1, 'objective': 'reg:squarederror'} # eta : learning rate
param['nthread'] = 4
param['eval_metric'] = 'rmse'

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, watchlist)

# make prediction
y_pred_train = bst.predict(dtrain)
y_pred_test = bst.predict(dtest)
print("Best Score (Train / Test) : {:.3f} / {:.3f}"
      .format(r2_score(y_train, y_pred_train),r2_score(y_test, y_pred_test)))

xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)

# Graphviz
import graphviz
dot = xgb.to_graphviz(bst, num_trees=1)
dot.render('xbg_tree')

# Dump, Save, Load 
bst.dump_model('xgb_model.txt') # bst.dump_model('dump.raw.txt', 'featmap.txt') 
bst.save_model('xgb_model.bin')
bst1 = xgb.Booster({'nthread': 4})  # init model
bst1.load_model('xgb_model.bin')  # load data

# PSO
from pyswarm import pso

score_train = []
score_test = []
x1 = []
x2 = []
x3 = []

def tuning_xgb(x):
    x1.append(np.int(x[0]))
    x2.append(np.int(x[1]))
    x3.append(x[2])
    param['max_depth'] = np.int(x[0])
    num_round = np.int(x[1])
    param['eta'] = x[2]

    bst = xgb.train(param, dtrain, num_round)
    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)
    score1 = r2_score(y_train, y_pred_train)
    score2 = r2_score(y_test, y_pred_test)

    score_train.append(score1)
    score_test.append(score2)
    print(f"Function Call : {len(x1)}, Objective : {-(score1 + score2)}")
    return -(score1 + score2)

lb = [1,  5, 0.1]
ub = [9, 30, 1]

xopt, fopt = pso(tuning_xgb, lb, ub, ieqcons=[], f_ieqcons=None, maxiter=1000, minstep=1e-8, minfunc=-200)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(score_train,score_test,s=3,alpha=0.2)
ax.set_title("PSO for XGBoost")
ax.set_xlabel('score_train')
ax.set_ylabel('score_test')
plt.axis('square')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()

param = {'max_depth': np.int(xopt[0]), 'eta': xopt[2]}
num_round = np.int(xopt[1])
bst = xgb.train(param, dtrain, num_round)
y_pred_train = bst.predict(dtrain)
y_pred_test = bst.predict(dtest)
print("Best Score (Train / Test) : {:.3f} / {:.3f}"
      .format(r2_score(y_train, y_pred_train),r2_score(y_test, y_pred_test)))
print("Best max_depth : {:.0f}\n"
      "Best num_round : {:.0f}\n"
      "Best eta : {:.3f}\n"
      .format(np.int(xopt[0]), np.int(xopt[1]), xopt[2]))

# Prediction Result
plt.figure()
plt.scatter(y_train, y_pred_train, c='b', s=10, alpha=0.5)
plt.scatter(y_test, y_pred_test, c='r', s=10, alpha=0.5)
plt.legend(["Train","Test"])
plt.title("XGBoost")
plt.xlabel("y")
plt.ylabel("Predicted y")
plt.axis('square')
plt.xlim(0.7,1.4)
plt.ylim(0.7,1.4)
plt.text(0.72, 1.2, f'R2 (Train / Test) = {r2_score(y_train, y_pred_train):.3f} / {r2_score(y_test, y_pred_test):.3f}' 
         , color='blue', fontsize=10)
plt.show()
