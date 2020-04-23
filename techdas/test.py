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
X = data_arr[:,2:44]
y = data_arr[:,45]

#####################################################
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.n_components_)

# plot the first two PCA dimensions
X_reduced = PCA(n_components=2).fit_transform(X)

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
           cmap=plt.cm.coolwarm, edgecolor='k', s=40)
ax.set_title("First two PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
legend1 = ax.legend(*scatter.legend_elements(num=5), loc="lower right", title="y value")
ax.add_artist(legend1)
'''
handles, labels = scatter.legend_elements(num=5, prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Size")
'''
plt.show()

# plot the first three PCA dimensions
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.coolwarm, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
handles, labels = scatter.legend_elements(num=5, prop="colors", alpha=0.6)
legend = ax.legend(handles, labels, loc="lower right", title="y value")
plt.show()

#####################################################
# PLS, SVM, NuSVM, SVM(RBF, Linear, Poly)
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 

fig, ax = plt.subplots(2,3,figsize=(15,10))

pls2 = PLSRegression(n_components=pca.n_components_)
pls2.fit(scale(X), y)
y_pred = pls2.predict(scale(X))
mse = mean_squared_error(y,y_pred)

ax[0,0].scatter(y, y_pred)
ax[0,0].set_title("PLS")
ax[0,0].set_xlabel("y")
ax[0,0].set_ylabel("Predicted y")
ax[0,0].axis([0,1.5,0,1.5])
ax[0,0].text(0.1, 0.1, r'MSE = 'f'{mse:.5f}', color='blue', fontsize=10)

clf0 = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
clf0.fit(scale(X), y)
y0_pred = clf0.predict(scale(X))
mse = mean_squared_error(y,y0_pred)

ax[0,1].scatter(y, y0_pred)
ax[0,1].set_title("SVR")
ax[0,1].set_xlabel("y")
ax[0,1].set_ylabel("Predicted y")
ax[0,1].axis([0,1.5,0,1.5])
ax[0,1].text(0.1, 0.1, r'MSE = 'f'{mse:.5f}', color='blue', fontsize=10)

clf1 = svm.NuSVR()
clf1.fit(scale(X), y)
y1_pred = clf1.predict(scale(X))
mse = mean_squared_error(y,y1_pred)

ax[0,2].scatter(y, y1_pred)
ax[0,2].set_title("NuSVR")
ax[0,2].set_xlabel("y")
ax[0,2].set_ylabel("Predicted y")
ax[0,2].axis([0,1.5,0,1.5])
ax[0,2].text(0.1, 0.1, r'MSE = 'f'{mse:.5f}', color='blue', fontsize=10)

clf2 = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
clf2.fit(scale(X), y)
y2_pred = clf2.predict(scale(X))
mse = mean_squared_error(y,y2_pred)

ax[1,0].scatter(y, y2_pred)
ax[1,0].set_title("SVM using RBF")
ax[1,0].set_xlabel("y")
ax[1,0].set_ylabel("Predicted y")
ax[1,0].axis([0,1.5,0,1.5])
ax[1,0].text(0.1, 0.1, r'MSE = 'f'{mse:.5f}', color='blue', fontsize=10)

clf3 = svm.SVR(kernel='linear', C=100, gamma='auto')
clf3.fit(scale(X), y)
y3_pred = clf3.predict(scale(X))
mse = mean_squared_error(y,y3_pred)

ax[1,1].scatter(y, y3_pred)
ax[1,1].set_title("SVM using Linear")
ax[1,1].set_xlabel("y")
ax[1,1].set_ylabel("Predicted y")
ax[1,1].axis([0,1.5,0,1.5])
ax[1,1].text(0.1, 0.1, r'MSE = 'f'{mse:.5f}', color='blue', fontsize=10)

clf4 = svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
clf4.fit(scale(X), y)
y4_pred = clf4.predict(scale(X))
mse = mean_squared_error(y,y4_pred)

#####################################################
# SVR with Grid Search
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn import svm

X_std = StandardScaler().fit_transform(X)
X_mas = MaxAbsScaler().fit_transform(X)

clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
scores_std = cross_val_score(clf, X_std, y, cv=5, scoring='neg_mean_squared_error')

clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
scores_mas = cross_val_score(clf, X_mas, y, cv=5, scoring='neg_mean_squared_error')

clf.get_params()

from sklearn.model_selection import GridSearchCV
from time import time

clf = svm.SVR(epsilon=.1)
param_grid = {'kernel':('poly', 'rbf'), 
              'C':np.linspace(1, 100, 5), 
              'gamma':np.power(10, np.arange(-1, 2, dtype=float))}
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error'))
start = time()
grid_search.fit(X_std, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

print("List of Scores :\n {0}"
      .format(grid_search.cv_results_['mean_test_score']))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report(grid_search.cv_results_)
'''
gamma는 가우시안 커널 폭의 역수 : 하나의 훈련 샘플이 미치는 영향의 범위를 결정
                                작은 값은 넓은 영역을 의미하며 훈련 샘플의 영향 범위도 커짐
                                --> 모델의 복잡도를 낮춤
C는 regularization factor : 증가시키면 결정 경계를 휘어서 정확하게 분류하게 함
'''
#####################################################
# Data Split, Normalization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train_scale = MaxAbsScaler().fit_transform(X_train)
X_test_scale = MaxAbsScaler().fit_transform(X_test)

#####################################################
# SVR Hyperparameter
n_iter = 1000
val_C = 1*np.random.uniform(1e-06, 1, n_iter)
val_gamma = 1
i = 0
score_train = []
score_test = []

while i < n_iter :
    model = svm.SVR(kernel='rbf', C=val_C[i], gamma=val_gamma)
    model.fit(X_train_scale, y_train)
    score_train.append(model.score(X_train_scale, y_train))
    score_test.append(model.score(X_test_scale, y_test))
    i = i+1

plt.scatter(val_C,score_train,s=2,alpha=0.2)
plt.scatter(val_C,score_test,s=2,alpha=0.2)
plt.xlabel('C')
plt.ylabel('R2 Score')
plt.legend(['Train','Test'])
plt.show()

val_C = 0.15
val_gamma = 10*np.random.uniform(1e-06, 1, n_iter)
i = 0
score_train = []
score_test = []

while i < n_iter :
    model = svm.SVR(kernel='rbf', C=val_C, gamma=val_gamma[i])
    model.fit(X_train_scale, y_train)
    score_train.append(model.score(X_train_scale, y_train))
    score_test.append(model.score(X_test_scale, y_test))
    i = i+1

plt.scatter(val_gamma,score_train,s=2,alpha=0.2)
plt.scatter(val_gamma,score_test,s=2,alpha=0.2)
plt.xlabel('gamma')
plt.ylabel('R2 Score')
plt.legend(['Train','Test'])
plt.show()

# PSO
from pyswarm import pso

score_train = []
score_test = []
x1 = []
x2 = []

def tuning(x):
    x_C     = x[0]
    x_gamma = x[1]
    x1.append(x_C)
    x2.append(x_gamma)
    model = svm.SVR(kernel='rbf', C=x_C, gamma=x_gamma)
    model.fit(X_train_scale, y_train)
    score1 = model.score(X_train_scale, y_train)
    score2 = model.score(X_test_scale, y_test)
    score_train.append(score1)
    score_test.append(score2)
    return -(score1 + score2)

lb = [1e-5, 1e-4]
ub = [1   , 10  ]

xopt, fopt = pso(tuning, lb, ub, ieqcons=[], f_ieqcons=None, maxiter=100, minstep=1e-8, minfunc=-200)

model = svm.SVR(kernel='rbf', C=xopt[0], gamma=xopt[1])
model.fit(X_train_scale, y_train)
print("Best Score (Train / Test) : {:.3f} / {:.3f}"
      .format(model.score(X_train_scale, y_train),model.score(X_test_scale, y_test)))
print("Best Hyperparameter (C / gamma) : {:.3f} / {:.3f}"
      .format(xopt[0], xopt[1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(score_train,score_test,s=3,alpha=0.2)
ax.set_xlabel('score_train')
ax.set_ylabel('score_test')
plt.show()
