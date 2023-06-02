import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from RTDB import get_rtdb_data

now = datetime.datetime.now()
print('Started : '+now.strftime('%Y-%m-%d %H:%M:%S'))

t1 = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
t2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
t1 = '2023-05-30 09:00:00'
t2 = '2023-06-01 09:00:00'

taglist=['712AI11015D1.PV','712AI11015F1.PV','712AI11015G1.PV','712AI11015H1.PV','712AI11015I1.PV',
         '712AI11015D2.PV','712AI11015F2.PV','712AI11015G2.PV','712AI11015H2.PV','712AI11015I2.PV',
         '712AI12015D1.PV','712AI12015F1.PV','712AI12015G1.PV','712AI12015H1.PV','712AI12015I1.PV',
         '712AI12015D2.PV','712AI12015F2.PV','712AI12015G2.PV','712AI12015H2.PV','712AI12015I2.PV',
         '712AI13015D1.PV','712AI13015F1.PV','712AI13015G1.PV','712AI13015H1.PV','712AI13015I1.PV',
         '712AI13015D2.PV','712AI13015F2.PV','712AI13015G2.PV','712AI13015H2.PV','712AI13015I2.PV',
         '712AI14015D1.PV','712AI14015F1.PV','712AI14015G1.PV','712AI14015H1.PV','712AI14015I1.PV',
         '712AI14015D2.PV','712AI14015F2.PV','712AI14015G2.PV','712AI14015H2.PV','712AI14015I2.PV',
         '712AI15015D1.PV','712AI15015F1.PV','712AI15015G1.PV','712AI15015H1.PV','712AI15015I1.PV',
         '712AI15015D2.PV','712AI15015F2.PV','712AI15015G2.PV','712AI15015H2.PV','712AI15015I2.PV']

df2 = get_rtdb_data(taglist, t1, t2, 1, 3600)
df2.index = pd.to_datetime(df2.index)

now = datetime.datetime.now()
print('RTDB Fetched : '+now.strftime('%Y-%m-%d %H:%M:%S'))

df2.plot()
plt.show()
