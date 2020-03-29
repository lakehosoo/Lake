import numpy as py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Downloading the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = ["age", "work-class", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
           
#data = pd.read_csv(url, names=columns, sep=',', na_values='?', skipinitialspace = True)
data = pd.read_csv('intro/data_analysis_with_pandas/adult.data', names=columns, sep=',', na_values='?', skipinitialspace = True)

######## Basic Analysis : Describe, Histogram, Count Plot 
data.head()
data.describe()
data.dtypes
# Age Statistics
print("Age Statistics")
print(data['age'].describe())
print("Median Age: ", data['age'].median())

# Plotting histogram for numerical values
numerical_attributes = data.select_dtypes(include=['int64'])
numerical_attributes.hist(figsize=(12,12))

# Plotting count plot for categorical values
categorical_attributes = data.select_dtypes(include=['object'])
plt.figure(figsize=(12,6))
sns.countplot(data = categorical_attributes, x = "work-class", hue = "income")

######## Data Quality
# Check missing values
data.info()
data.isnull().sum()
#print("Missing value count occupation: ", data['occupation'].isnull().sum())

# 1. Drop these rows, so that we don't have any missing values in our data.
# 2. Choose the median of values once we have transformed categorical values into corresponding numerical representations, as median is not affected by the range/spread of data unlike mean.
# 3. We could also use a classifier with non-missing values to predict the missing values, and then use them to build our final model.
# 4. There are certain classifiers like XGBoost which automatically handle missing data, but we won't assume that here.

######## Data Processing
# Dropping missing values
data = data.dropna()
# Dropping duplicate values
data = data.drop_duplicates()
# Dropping outlier
i = data[data['capital-gain'] > 80000].index
data = data.drop(i)
# Dropping the column fnlwgt
data = data.drop(columns='fnlwgt')
# Combining capital-gain and capital-loss into 1 column
data['netcapitalgain'] = data.apply(lambda x: x['capital-gain'] + x['capital-loss'], axis=1)
data = data.drop(columns='capital-gain')
data = data.drop(columns='capital-loss')

data.info()

######## Data Relationships
# Pair Plot : kde(Kernel densith estimation)
sns.pairplot(data, height=2, diag_kind = 'kde', hue='income')

# Correlation Heatmap
corr = data.corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 12))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax = ax, annot=True, linewidth = 0.1)
plt.title('Pearson Correlation of Features', y=1.05, size=15)

# Cross tabulation between work-class and sex
pd.crosstab(data['work-class'],data['sex'], margins=True)

# Box plot between work-class and age for different sex
plt.figure(figsize=(12,6))
sns.boxplot(x="work-class",y="age", hue="sex", data=data)

# plot Swarmplot + Box
plt.figure(figsize=(12,6))
sns.swarmplot(data=data, x="work-class",y="age", hue="sex") # --> Too Many
sns.boxplot(data=data, x="work-class",y="age", hue="sex",  
            showcaps=True,                 # 박스 상단 가로 라인
            whiskerprops={'linewidth':1},  # 박스 상단 세로 라인 
            showfliers=True,               # 아웃라이어 표시
            boxprops={'facecolor':'None'}, # 박스 색상
        )

