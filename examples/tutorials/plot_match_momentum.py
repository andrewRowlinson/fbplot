"""
===============
Match Momentum
===============

This example shows how to plot match momentum using the calculations of expected threat (xT) model shown in the same folder.
It can be applied to any match where we have the xT values of every event that happend, the team that made it and the minute when it happened.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

##############################################################################
# Import the data
# ---------------
# For this, we're using data from the FA Women's Super League 2019/20 as used in the `Improvements to expected threat <https://mplsoccer.readthedocs.io/en/latest/gallery/tutorials/plot_xt_improvements.html>`
# but just limiting to one match with the xT already calculated (dataframe move_success).
# The only thing that changes is the fact that we need the columns of "minute" and "team_name" to plot match momentum correctly.
# You may add them where the list "cols" is defined in the other tutorial.

match_events = pd.read_csv("https://raw.githubusercontent.com/andrewRowlinson/mplsoccer-assets/main/xt_match_example.csv")

##############################################################################
# Calculating the moving averages of Expected Threat
# --------------
# Here we need to know the threat that every team had in each minute of the match
# For that, we need to split the match_events dataframe in two separate ones (one for each team) and group the sum of xT values
# by minutes.

teams = match_events.team_name.unique()
team1, team2 = teams[0], teams[1]

team1_actions = match_events[match_events['team_name'] == team1].groupby('minute').sum()
team2_actions = match_events[match_events['team_name'] == team2].groupby('minute').sum()

# With the two dataframes separated, we need to calculate the mean of the moving average of the Expected Threat.
# We are using five as a parameter to take into consideration the last five minutes of the match.

team1_actions['rolling_xt'] = team1_actions.xt.rolling(5).mean()
team2_actions['rolling_xt'] = team2_actions.xt.rolling(5).mean()

# Then we need to merge them by the minute so we can calculate the difference between the two values
# which is going to give us the values of the y-axis in the match momentum plot.

match_xt = team1_actions.merge(team2_actions, left_index=True, right_index=True, how='outer', suffixes=['_t1', '_t2'])
match_xt = match_xt[['rolling_xt_t1', 'rolling_xt_t2']].fillna(0) #so the difference can be calculated correctly.
match_xt['total_diff'] = match_xt['rolling_xt_t1'] - match_xt['rolling_xt_t2']

##############################################################################
# Plot Match Momentum
# --------------
# The only things we need from the events are the minutes (x-axis) and the difference between the moving averages of xT for both teams (y-axis)
# Also, is needed that we interpolate the values.

fig, ax = plt.subplots(figsize=(16,9))
fig.set_facecolor('white')

local_color = 'lightblue'
visit_color = 'red'

x = match_xt.index
y = match_xt.total_diff

X_Y_Spline = make_interp_spline(x, y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)

ax.plot(X_, Y_)

# Then all esthetic functions for the plot to be more readable and clear

ax.axhline(0) # To separate the values

# To fill with one color for the positive and negative values we use this function

ax.fill_between(X_, Y_, color=local_color, where= Y_ > 0)
ax.fill_between(X_, Y_, color=visit_color, where= Y_ < 0)

ax.set_xlim(0,match_xt.index[-1]) #To compact the plot in the x-axis
ax.spines[['right','left','top']].set_visible(False)

# Considerations in x and y axis.
ax.set_xlabel('Minutes', fontsize=20)
ax.set_yticks([]) #You can comment this line if you want the difference of values to be shown
ax.set_xticks(range(0,match_xt.index[-1], 10))

# Lastly, if we have the data of goals in out events we can make to lists: one for the minutes and another with which team scored.
# With that we can plot a line showing every single one of them

min_goals = [ 6,  9, 16, 22, 69]
team_goals = ['Reading WFC', 'Everton LFC', 'Reading WFC', 'Everton LFC', 'Reading WFC']


for goal in range(len(min_goals)):
    if team_goals[goal] == team1:
        ax.axvline(min_goals[goal], lw=2, color=local_color, ls=':')
    else:
        ax.axvline(min_goals[goal], lw=2, color=visit_color, ls=':')

#And a title using the data of the team_name columns
ax.set_title(f'Match momentum {team1} vs. {team2}', fontsize=26)

plt.show()