import pandas as pd
import matplotlib.pyplot as plt

# stock ticker symbol
url = 'http://apmonitor.com/che263/uploads/Main/goog.csv'

# import data with pandas
mydata = pd.read_csv(url)
print(mydata['Close'][0:5])
print('min: '+str(min(mydata['Close'][0:20])))
print('max: '+str(max(mydata['Close'][0:20])))

# plot data with pyplot
plt.figure()
plt.plot(mydata['Open'][0:20], label='Open')
plt.plot(mydata['Close'][0:20], label='Close')
plt.xlabel('days ago')
plt.ylabel('price')
plt.legend()
plt.show()
