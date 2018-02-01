# coding:utf-8
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import argparse
import numpy as np
def get_better_model(model_name):
    df = pd.read_csv('test%s.csv'%(model_name))
    s = df.columns[1]
    g = df.columns[3]

    score = np.array(df[s])[:num_model*num_test].astype(float)
    goal_time = np.array(df[g])[:num_model*num_test]

    score = score.reshape((num_model,num_test))
    goal_time = goal_time.reshape((num_model,num_test))

    clear_count_list = []
    better_model_index = []

    with open("./betterModelLog_A.csv", 'a') as the_file:
        the_file.write("%s,,\nModel,Clear Count,AverageScore,MaxScore,MinScore,AverageTime,FastestTime,SlowestTime\n"%(model_name))

    for i in range(num_model):
        clear_count = np.sum(goal_time[i] > 0)
        clear_count_list.append(clear_count)

        if(clear_count>=40):
            better_model_index.append(i)
            with open("./betterModelLog_A.csv", 'a') as the_file:
                the_file.write(str((i+1)*10000) +
                           ',' + str(clear_count) +
                           ',' + str(np.average(score[i])) +
                           ',' + str(np.max(score[i])) +
                           ',' + str(np.min(score[i])) +
                           ',' + str(np.average(goal_time[i])) +
                           ',' + str(np.min(goal_time[i][goal_time[i]>0])) +
                           ',' + str(np.max(goal_time[i])) + '\n')


    return clear_count_list

with open("./betterModelLog_A.csv", 'w') as the_file:
    the_file.write("Better Models\n")

num_model = 60
num_test = 50
#model_name_list = ["Action3","Action5","Action7_1","Action7_2"]
#model_name_list = ["Action3V","Action5V","Action7_1V","Action7_2V"]
#model_name_list = ["Action3_fix","Action3V_fix"]
model_name_list = ["Action3A","Action3VA"]

print "Number of Model : ",num_model
print "Number of Test : ",num_test

x = range(10000,num_model*10**4+1,10000)

plt.xticks(range(0,num_model*10**4+1,num_model*10**4/5))
plt.xlabel("Cycle") # x軸のラベル
plt.ylabel("Number of goal") # y軸のラベル

for model_name in model_name_list:
    ccl = get_better_model(model_name)

    if(len(model_name_list)==4):
        if(model_name == model_name_list[0]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model1V', color='red', linewidth=2.5)
        elif(model_name == model_name_list[1]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model2V', color='green', linewidth=2.5)
        elif(model_name == model_name_list[2]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model3V', color='blue', linewidth=2.5)
        elif(model_name == model_name_list[3]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model4V', color='#ffc700', linewidth=2.5)

    elif(len(model_name_list)==2):
        if(model_name == model_name_list[0]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model1_forward', color='red', linewidth=2.5)
        elif(model_name == model_name_list[1]):
            plt.plot(x[:num_model], ccl[:num_model], label='Model1V_forward', color='green', linewidth=2.5)

    plt.legend(loc = 'upper left') #これをしないと凡例出てこない(lower⇆upper, left⇆ center ⇆right)
    plt.show()


'''
df = pd.read_csv('testAction7_2.csv')
s = df.columns[1]
g = df.columns[3]

score = np.array(df[s])[:1000].astype(float)
goal_time = np.array(df[g])[:1000]

score[29]
goal_time[29]
len(score)
num_model = score.shape[0]/10
num_model
score = score.reshape((num_model,10))
goal_time = goal_time.reshape((num_model,10))

df_b = pd.read_csv('testAction7_2_better.csv')
s_b = df.columns[1]
g_b = df.columns[3]

score_b = np.array(df_b[s_b])[:1000]
goal_time_b = np.array(df_b[g_b])[:1000]

print score_b[28]
print goal_time_b[28]

print len(score_b)

num_model_b = score_b.shape[0]/10
print num_model_b
score_b = score_b.reshape((num_model_b,10))
goal_time_b = goal_time_b.reshape((num_model_b,10))


print score[64]
print goal_time[64]
print score_b[10]
print goal_time_b[10]
clear_count_list = []

#with open("bestModelLog.csv", 'w') as the_file:
    #the_file.write("Acion3,,\nModel,AverageTime,FastestTime\n")
j = 0
for i in range(score.shape[0]):
    clear_count = np.sum(goal_time[i] > 0)
    clear_count_list.append(clear_count)
    if(clear_count==10):
        print i
        score[i] = score_b[j].copy()
        goal_time[i] = goal_time_b[j].copy()
        j+=1
        #logファイルへの書き込み
        #with open("bestModelLog.csv", 'a') as the_file:
            #the_file.write(str((i+1)*10000) +
                       #',' + str(np.average(score[i])) +
                       #',' + str(np.min(score[i])) + '\n')

score = score.ravel()
goal_time = goal_time.ravel()
print score.shape

with open("hoge7_2.csv", 'w') as the_file:
                the_file.write('Cycle,Score,Episode,GoalTime \n')
for i in range(1000):
    with open("hoge7_2.csv", 'a') as the_file:
                    the_file.write(str(i+1) +
                               ',' + str(score[i]) +
                               ',' + str(i+1) +
                               ',' + str(goal_time[i]) + '\n')
'''
