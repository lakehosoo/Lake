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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_tot, y_tot, random_state=1)
X_train_scale = MaxAbsScaler().fit_transform(X_train)
X_test_scale = MaxAbsScaler().fit_transform(X_test)
#X_train_scale = StandardScaler().fit_transform(X_train)
#X_test_scale = StandardScaler().fit_transform(X_test)

#####################################################
# NN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlp = MLPRegressor(solver='lbfgs', random_state=0, alpha=0.001, hidden_layer_sizes=[128,64]) # activation='tanh', hidden_layer_sizes=[10, 10]
mlp.fit(X_train_scale, y_train)
print("Score of Train / Test : {:.3f} / {:.3f}".format(mlp.score(X_train_scale, y_train),mlp.score(X_test_scale, y_test)))

# PSO
from pyswarm import pso

score_train = []
score_test = []
x1 = []
x2 = []
x3 = []

def tuning_mlp(x):
    x1.append(x[0])
    x2.append(np.int(x[1]))
    x2.append(np.int(x[2]))
    
    mlp = MLPRegressor(solver='lbfgs', random_state=0, max_iter=1000,
                       alpha=x[0], hidden_layer_sizes=[np.int(x[1]),np.int(x[2])]) 
    mlp.fit(X_train_scale, y_train)
    score1 = mlp.score(X_train_scale, y_train)
    score2 = mlp.score(X_test_scale, y_test)

    score_train.append(score1)
    score_test.append(score2)
    print(f"Function Call : {len(x1)}, Objective : {-(score1 + score2)}")
    return -(score1 + score2)

lb = [0.0001,  10,  10]
ub = [     1, 128, 128]

xopt, fopt = pso(tuning_mlp, lb, ub, ieqcons=[], f_ieqcons=None, maxiter=10, minstep=1e-8, minfunc=-200)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(score_train,score_test,s=3,alpha=0.2)
ax.set_title("PSO for NN")
ax.set_xlabel('score_train')
ax.set_ylabel('score_test')
plt.axis('square')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()


# Prediction Result
y_pred_train = mlp.predict(X_train_scale)
y_pred_test = bmlp.predict(X_test_scale)

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
plt.text(0.72, 1.2, f'R2 (Train / Test) = {mlp.score(X_train_scale, y_train):.3f} / {mlp.score(X_test_scale, y_test):.3f}' 
         , color='blue', fontsize=10)
plt.show()
