import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data_value = pd.read_excel('HME_Data_V2.xlsx', sheet_name='Rearrange', index_col=0, header=None, skiprows=2).transpose()
#data_index = pd.read_excel('HME_Data_V2.xlsx', sheet_name='Rearrange', index_col=0, header=None,nrows=2, dtype=object).transpose()
data_value = pd.read_csv('data', index_col=0, header=None, skiprows=2).transpose()
data_index = pd.read_csv('data', index_col=0, header=None,nrows=2, dtype=object).transpose()
data_index.columns=["Name","Grade"]

# Grade Information
for i in range(data_index.shape[0]):
    if data_index.iloc[i,1]==0:
        data_index.iloc[i,1]=data_index.iloc[i-1,1]

data = pd.concat([data_index,data_value],axis=1)
data.describe()
data.info()

#data.hist(figsize=(20, 15))
#pd.scatter_matrix(data, diagonal='kde', color='b', alpha=0.3, figsize=(20, 15))
data['Grade'].value_counts()
data['Grade'].hist()
plt.scatter(data['Prop1'],data['Prop15'])

'''
# Drop the Unused Input
for i in range(43): 
    if data['Input'+f'{i+1}'].sum()==0:
        data = data.drop(columns='Input'+f'{i+1}')
'''
'''
# Simple Average of Each Properties 
for i in range(18): 
    if data['Prop'+f'{i+1}'].mean()
        data = data.drop(columns='Prop'+f'{i+1}')
'''
# New Output
name = []
for i in range(data_index.shape[0]):
    if (name != data['Name'][i+1]):
        name = data['Name'][i+1]
        if data['Prop15'][i+1]:
            Prop_base = data['Prop15'].iloc[i]
        elif data['Prop17'][i+1]:
            Prop_base = data['Prop17'].iloc[i]
        data['Prop_New'].iloc[i] = 0
    else:
        if data['Prop15'].iloc[i] > 0:
            if data['Prop15'].iloc[i] > Prop_base:
                data['Prop_New'].iloc[i] = 1
            else:
                data['Prop_New'].iloc[i] = -1
        elif data['Prop17'][i+1] > 0:
            if data['Prop17'][i+1] > Prop_base:
                data['Prop_New'].iloc[i] = 1
            else:
                data['Prop_New'].iloc[i] = -1
        else:
            data['Prop_New'].iloc[i] = np.nan

#data.apply(lambda x: x['Grade']=='REF1', axis=1)
#data5 = data[data['Grade']=='REF5']
#data[['Name','Prop15','Prop17','Prop_New']][0:20]
i = data[data['Prop_New'].isnull() == True].index
data_drop = data.drop(i)

data_drop.shape
sns.set(style="darkgrid")
ax = sns.countplot(x="Prop_New", hue="Grade", data=data_drop)

#data[data['Grade']=='REF5'],
#data.hist(figsize=(20, 15))
#pd.scatter_matrix(data, diagonal='kde', color='b', alpha=0.3, figsize=(20, 15))
