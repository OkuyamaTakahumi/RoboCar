# coding:utf-8
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import argparse
import numpy as np
print 3

df1 = pd.read_csv('rewardAction3_Real.csv')
#df2 = pd.read_csv('rewardAction3V_Real.csv')

x1 = df1.columns[0]
y1 = df1.columns[1]
#x2 = df2.columns[0]
#y2 = df2.columns[1]



df1[y1] = pd.rolling_mean(df1[y1], window=10)
#df2[y2] = pd.rolling_mean(df2[y2], window=100)

y1_array = np.array(df1[y1])
#y2_array = np.array(df2[y2])

fig, ax = plt.subplots(1, 1)
plt.xticks(range(0,30001,5000))

plt.xlabel("Cycle") # x軸のラベル
plt.ylabel("Score") # y軸のラベル

plt.plot(list(df1[x1]), y1_array, label='Model1_Real', color='red', linewidth=2.5)
#plt.plot(list(df2[x2]), y2_array, label='Model1V_Real', color='green', linewidth=2.5)


plt.legend(loc = 'upper left') #これをしないと凡例出てこない(lower⇆upper, left⇆ center ⇆right)
plt.show()

'''
cycle_array = np.array(df1[x1])
cycle_array.shape
y1_array.shape

last_cycle = 0
cycle_score = []
for i in range(len(cycle_array)):
    cycle_score.append(cycle_array[i] - last_cycle)
    last_cycle = cycle_array[i]
max(cycle_score)
plt.plot(list(df1[x1]), cycle_score, label='Model1_Real', color='red', linewidth=2.5)
'''
max(cycle_score)
np.max(y1_array)
