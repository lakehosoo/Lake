import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
df = pd.read_csv('intro/data_analysis_with_pandas/automobile.csv')

df.head()
df.describe()

df['num-of-doors'].value_counts()

sns.boxplot(x='num-of-cylinders',y='price',data=df)

plt.scatter(df['engine-size'],df['price'])
plt.xlabel('Engine Size')
plt.ylabel('Price')

count,bin_edges = np.histogram(df['peak-rpm'])
df['peak-rpm'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Value of peak rpm')
plt.ylabel('Number of cars')
plt.grid()

df_temp = df[['num-of-doors','body-style','price']]
df_group = df_temp.groupby(['num-of-doors','body-style'],as_index=False).mean()

# ANOVA : Analysis of Variance
# F-test score: It calculates the variation between sample group means divided by variation within sample group.
# P value: It shows us the confidence degree. In other words, it tells us whether the obtained result is statistically significant or not.
temp_df = df[['make','price']].groupby(['make'])
stats.f_oneway(temp_df.get_group('audi')['price'],temp_df.get_group('volvo')['price'])
stats.f_oneway(temp_df.get_group('alfa-romero')['price'],temp_df.get_group('peugot')['price'])

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)

sns.regplot(x='engine-size',y='price',data=df)