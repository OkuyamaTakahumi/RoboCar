# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import argparse

'''
logファイルの中身
0列目 cycle
1列目 score
2列目 episode_num
3列目 goal_time
'''

parser = argparse.ArgumentParser()
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help='reward log file name')
parser.add_argument('--x_axis', '-x', default='cycle', type=str,
                    help='X_axis Value : cycle or episode')
parser.add_argument('--y_axis', '-y', default='score', type=str,
                    help='Y_axis Value : score or goal')


args = parser.parse_args()

df = pd.read_csv(args.log_file)

if(args.x_axis == "cycle"):
    x = df.columns[0]
elif(args.x_axis == "episode"):
    x = df.columns[2]
else:
    print u"正しいX-axisのParserを指定してください"
    import sys
    sys.exit()

if(args.y_axis == "score"):
    y = df.columns[1]
elif(args.y_axis == "goal"):
    y = df.columns[3]
else:
    print u"正しいY-axisのParserを指定してください"
    import sys
    sys.exit()

ax = df.plot(kind='scatter', x=x, y=y)

df[y] = pd.rolling_mean(df[y], window=20)
df.plot(kind='line', x=x, y=y, ax=ax)
plt.show()
