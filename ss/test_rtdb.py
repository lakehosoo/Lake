from RTDB import get_rtdb_data
from PHD import fetch_rtdb_data
import pandas as pd
import matplotlib.pyplot as plt

res_=[]
with open("ss_main_output.txt", 'r') as f:
    res = f.readlines()

df_raw=pd.DataFrame([i.split('\n', 1)[0] for i in res])
df=pd.DataFrame(df_raw[0].str.split(',').tolist())
df.index = pd.to_datetime(df[0])
df.drop(columns=0, inplace=True)
df = df.astype('float')
df.iloc[:,2:]=df.iloc[:,2:]*100

t1=df.index.min().strftime("%Y-%m-%d %H:%M:%S")
t2=df.index.max().strftime("%Y-%m-%d %H:%M:%S")

taglist=['712AI11015D1.PV','712AI11015F1.PV','712AI11015G1.PV','712AI11015H1.PV','712AI11015I1.PV',
         '712AI11015D2.PV','712AI11015F2.PV','712AI11015G2.PV','712AI11015H2.PV','712AI11015I2.PV',
         '712AI12015D1.PV','712AI12015F1.PV','712AI12015G1.PV','712AI12015H1.PV','712AI12015I1.PV',
         '712AI12015D2.PV','712AI12015F2.PV','712AI12015G2.PV','712AI12015H2.PV','712AI12015I2.PV']
df2 = fetch_rtdb_data(taglist, t1, t2, freq="1200", method="snapshot")
df2.index = pd.to_datetime(df2.index)

taglis2=['712TC11100.PV','712TC11300.PV','712TC11500.PV','712TC11700.PV',
         '712TC11200.PV','712TC11400.PV','712TC11600.PV','712TC11800.PV',
         '712TC12100.PV','712TC12300.PV','712TC12500.PV','712TC12700.PV',
         '712TC12200.PV','712TC12400.PV','712TC12600.PV','712TC12800.PV']
df3 = fetch_rtdb_data(taglis2, t1, t2, freq="1200", method="snapshot")
df3.index = pd.to_datetime(df3.index)

lim=[[0,20],[0,30],[0,35],[0,5],[0,20]]
fn=[[111,1],[111,2],[121,1],[121,2]]
fig, axes = plt.subplots(figsize=(25,20), nrows=4, ncols=5)
for k in range(0,4):
    for i in range(0,5):
        df[(df[1]==fn[k][0]) & (df[2]==fn[k][1])][3+i].plot(ax = axes[k,i], linestyle='',linewidth=1, c='m', marker='o', ms=3, xlabel='')
        df[(df[1]==fn[k][0]) & (df[2]==fn[k][1])][8+i].plot(ax = axes[k,i], linestyle='',linewidth=1, c='r', marker='o', ms=3, xlabel='')
        df2[taglist[5*k+i]].plot(ax = axes[k,i], linestyle='',linewidth=1, c='b', marker='o', ms=2)
        axes[k,i].set_ylim(lim[i])
        
        ax2 = axes[k,i].twinx()
        df3.iloc[:,4*k:4*(k+1)].mean(axis=1).plot(ax = ax2, linestyle='',linewidth=1, c='grey', marker='o', ms=3)
        ax2.set_ylim(840,850)

plt.tight_layout()
