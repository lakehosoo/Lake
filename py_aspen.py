import os
import numpy as np
import matplotlib.pyplot as plt
import win32com.client as win32
aspen = win32.Dispatch('Apwn.Document')

filepath=os.path.abspath(r'C:\Users\lakehosoo\py_aspen\model\Flash_Example.bkp')
aspen.InitFromArchive2(filepath)
#aspen.Visible=True

## Test Variables
feed_temp = aspen.Tree.FindNode('\Data\Streams\FEED\Input\TEMP\MIXED').Value
flsh_temp = aspen.Tree.FindNode('\Data\Blocks\FLASH\Input\TEMP').Value

print("Feed temperature = {0} degC".format(feed_temp))
print("Flash temperature = %s" % flsh_temp)

## Sensitivity Analysis
T = np.linspace(70, 100, 20)

x_ethanol, y_ethanol = [], []

for temperature in T:
    aspen.Tree.FindNode('\Data\Blocks\FLASH\Input\TEMP').Value = temperature
    aspen.Engine.Run2()

    x_ethanol.append(aspen.Tree.FindNode('\Data\Streams\LIQUID\Output\MOLEFRAC\MIXED\ETHANOL').Value)
    y_ethanol.append(aspen.Tree.FindNode('\Data\Streams\VAPOR\Output\MOLEFRAC\MIXED\ETHANOL').Value)

plt.plot(T, y_ethanol, 'ro', T, x_ethanol, 'bo')
plt.plot(T, y_ethanol, 'r-', T, x_ethanol, 'b-')
plt.legend(['vapor', 'liquid'])
plt.xlabel('Flash Temperature (degC)')
plt.ylabel('Ethanol mole fraction')
#plt.show()  
plt.savefig('py_aspen/Flash_Temperature_fig_1.png')

## Close Function is Not Working
aspen.Close()
