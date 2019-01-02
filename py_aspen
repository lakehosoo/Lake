import os
import win32com.client as win32
aspen = win32.Dispatch('Apwn.Document')

filepath=os.path.abspath(r'model\Flash_Example.bkp')
aspen.InitFromArchive2(filepath)

## Input variables
feed_temp = aspen.Tree.FindNode('\Data\Streams\FEED\Input\TEMP\MIXED').Value
print("Feed temperature = {0} degC".format(feed_temp))

flsh_temp = aspen.Tree.FindNode('\Data\Blocks\FLASH\Input\TEMP').Value
print("Flash temperature = %s" % flsh_temp)

## Close Function is Not Working
aspen.Close()
