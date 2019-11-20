# Dynamic Plotting Tool for R Plant PIS Data 
# By Hosoo Kim, LG Chem
# Rev. Date: 2019.10.24

# import
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as matcolors
import pandas as pd
from matplotlib.widgets import Slider

# Loading the data file =============================================================
data_file = pd.read_csv('profile_data_1002_v2.csv')
ti_index = pd.read_excel('ti_map.xlsx',sheet_name='index', header=0)#, index_col ='index')
ti_map1 = pd.read_excel('ti_map.xlsx',sheet_name='tag1', header=0)#, index_col ='tag')
ti_map2 = pd.read_excel('ti_map.xlsx',sheet_name='tag2', header=0)#, index_col ='tag')

time = pd.to_datetime(data_file.iloc[1:,0],yearfirst=True)
time_max = time.count()

for i in ti_map1.index:
    ti_map1.loc[i,'x'] = ti_index[ti_index['index']==ti_map1.loc[i,'index']]['radius'].values \
                         *np.cos(np.deg2rad(ti_index[ti_index['index']==ti_map1.loc[i,'index']]['theta'])).values
    ti_map1.loc[i,'y'] = ti_index[ti_index['index']==ti_map1.loc[i,'index']]['radius'].values \
                         *np.sin(np.deg2rad(ti_index[ti_index['index']==ti_map1.loc[i,'index']]['theta'])).values

for i in ti_map2.index:
    ti_map2.loc[i,'x'] = ti_index[ti_index['index']==ti_map2.loc[i,'index']]['radius'].values \
                         *np.cos(np.deg2rad(ti_index[ti_index['index']==ti_map2.loc[i,'index']]['theta'])).values
    ti_map2.loc[i,'y'] = ti_index[ti_index['index']==ti_map2.loc[i,'index']]['radius'].values \
                         *np.sin(np.deg2rad(ti_index[ti_index['index']==ti_map2.loc[i,'index']]['theta'])).values

#pf_101 = pd.concat([time,data_file.iloc[1:,10:64]],axis=1)
#pf_102 = pd.concat([time,data_file.iloc[1:,64:118]],axis=1)
pf_101 = data_file.iloc[1:,10:64]
pf_102 = data_file.iloc[1:,64:118]

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d', position=[0.05, 0.1, 0.4, 0.8])
ax2 = fig.add_subplot(122, projection='3d', position=[0.45, 0.1, 0.4, 0.8])

# fig, (ax1, ax2, cax) = plt.subplots(ncols=3, figsize=(12,6), gridspec_kw={'width_ratios': [1,1,0.1]})
# ax1 = fig.subplots(131, projection='3d')
# ax2 = fig.subplots(132, projection='3d')
# cax = fig.subplots(133)

# norm = matcolors.Normalize(vmin=min(min(pf_101.min()),min(pf_102.min())), \
#                            vmax=max(max(pf_101.max()),max(pf_102.max())) )
norm = matcolors.Normalize(vmin=360, vmax=405)

colors1 = pf_101.iloc[0]
graph1 = ax1.scatter(ti_map1.x, ti_map1.y, ti_map1.height, c=colors1, s=40, cmap="coolwarm", norm=norm, edgecolors='none')
graph1.set_cmap("coolwarm")
graph1.set_norm(norm)
ax1.set_zlabel('Height [mm]',labelpad=10)
ax1.view_init(elev=75, azim=-62)

colors2 = pf_102.iloc[0]
graph2 = ax2.scatter(ti_map2.x, ti_map2.y, ti_map2.height, c=colors2, s=40, cmap="coolwarm", norm=norm, edgecolors='none')
graph2.set_cmap("coolwarm")
graph2.set_norm(norm)
ax2.set_zlabel('Height [mm]',labelpad=10)
ax2.view_init(elev=75, azim=-62)

# Cylinder
x = np.linspace(-1, 1, 50)
z = np.linspace(0, 3400, 100)
Xc, Zc = np.meshgrid(x, z)
Yc = np.sqrt(1-Xc**2)
# Draw parameters
rstride = 20
cstride = 10
ax1.plot_surface(Xc, Yc, Zc, alpha=0.1, color="gray", rstride=rstride, cstride=cstride)
ax1.plot_surface(Xc, -Yc, Zc, alpha=0.1, color="gray", rstride=rstride, cstride=cstride)
ax2.plot_surface(Xc, Yc, Zc, alpha=0.1, color="gray", rstride=rstride, cstride=cstride)
ax2.plot_surface(Xc, -Yc, Zc, alpha=0.1, color="gray", rstride=rstride, cstride=cstride)

cax = fig.add_axes([0.92, 0.1, 0.02, 0.7])
cbar = plt.colorbar(graph2, cax=cax)
cbar.set_label('Temperature [C]')

axcolor = 'lightgoldenrodyellow'
axtime1 = plt.axes([0.10, 0.05, 0.3, 0.03], facecolor=axcolor)
axtime2 = plt.axes([0.50, 0.05, 0.3, 0.03], facecolor=axcolor)
# Slider
stime1 = Slider(axtime1, 'Time', 1, time_max, valinit=1, valstep=1)
stime2 = Slider(axtime2, 'Time', 1, time_max, valinit=1, valstep=1)

#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

def update(val):
    # t1 is the current value of the slider
    t1 = int(stime1.val)
    t2 = int(stime2.val)
    # update title
    label1 = '101-D at '+ str(time[t1])
    label2 = '102-D at '+ str(time[t2])
    ax1.set_title(label1)
    ax2.set_title(label2)
    # update curve
    graph1.set_array(pf_101.iloc[t1])
    graph2.set_array(pf_102.iloc[t2])
    # redraw canvas while idle
    fig.canvas.draw_idle()

# call update function on slider value change
stime1.on_changed(update)
stime2.on_changed(update)

plt.show()
