##### Modeling Time Series #####
import pandas as pd
import statsmodels.api as sm

store = 'data/ts.hdf5'
df = pd.read_hdf(store, 'ts')

with pd.option_context('display.max_rows', 100):
    print(df.dtypes)

daily = df.fl_date.value_counts().sort_index()
y = daily.resample('MS').mean()
y.head()