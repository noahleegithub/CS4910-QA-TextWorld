import ast
lines = []
with open('results.txt') as f:
    lines = f.readlines()

replace = ['array','(',')']
count = 0
qa_acc = 0
suff_acc = 0
mean_cum_reward =[]
episodes = []
games = 0
for line in lines:
    count += 1
    episodes.append(count)
    for r in replace : line = line.replace(r,'')
    mp = ast.literal_eval(line)
    games = len(mp['qa_correct'])
    for b in mp['qa_correct'] :
        if b: qa_acc += 1 
    for b in mp['suff_info_correct']: 
        if b: suff_acc += 1
    mean_cum_reward.append(mp['cumulative_rewards_mean'])
    # print(f'line {count}: {mp}')
qa_acc /= count * games
suff_acc /= count * games
print(f'qa_acc: {qa_acc} over {count} episodes')
print(f'suff_acc: {suff_acc} over {count} episodes')


# plot stuff
import matplotlib.pyplot as plt
import seaborn as sns

# python suggestion to eliminate conversion and register errors
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set()
plt.xlabel('epsiode')
plt.ylabel('mean cumulative reward over games')
plt.title('cumulative rewards')

plt.plot(episodes,mean_cum_reward,linewidth=3)
plt.savefig('plot.png')


# notes 
# Random agent
# qa_correct : true if the agent answered correctly
  # what room is the stove in?
# suff_info_correct : true if the agent stops in a good final position
  # where is the knife
  # reward is given if they stop in kitchen
# y : mean cumulative reward(final for both summed / 2), x = episodes