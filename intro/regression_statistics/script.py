import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# pip install uncertainties, if needed
try:
     import uncertainties.unumpy as unp
     import uncertainties as unc
except:
     import pip
     pip.main(['install','uncertainties'])
     import uncertainties.unumpy as unp
     import uncertainties as unc

# import data
 url = 'http://apmonitor.com/che263/uploads/Main/stats_data.txt'
 data = pd.read_csv(url)
 x = data['x'].values
 y = data['y'].values
 n = len(y)



 import matplotlib.pyplot as plt 
