"""
=====================
Pass plot using lines
=====================

This example shows how to plot all passes from a team in a match as lines.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen

rcParams['text.color'] = '#c7d5cc'  # set the default text color

# get event dataframe for game 7478
parser = Sbopen()
df, related, freeze, tactics = parser.event(7478)

##############################################################################
# Boolean mask for filtering the dataset by team

team1, team2 = df.team_name.unique()
mask_team1 = (df.type_name == 'Pass') & (df.team_name == team1)

##############################################################################
# Filter dataset to only include one teams passes and get boolean mask for the completed passes

df_pass = df.loc[mask_team1, ['x', 'y', 'end_x', 'end_y', 'outcome_name']]
mask_complete = df_pass.outcome_name.isnull()

##############################################################################
# View the pass dataframe.

df_pass.head()

##############################################################################
# Plotting

# Setup the pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
fig.set_facecolor('#22312b')

# Plot the completed passes
lc1 = pitch.lines(df_pass[mask_complete].x, df_pass[mask_complete].y,
                  df_pass[mask_complete].end_x, df_pass[mask_complete].end_y,
                  lw=5, transparent=True, comet=True, label='completed passes',
                  color='#ad993c', ax=ax)

# Plot the other passes
lc2 = pitch.lines(df_pass[~mask_complete].x, df_pass[~mask_complete].y,
                  df_pass[~mask_complete].end_x, df_pass[~mask_complete].end_y,
                  lw=5, transparent=True, comet=True, label='other passes',
                  color='#ba4f45', ax=ax)

# Plot the legend
ax.legend(facecolor='#22312b', edgecolor='None', fontsize=20, loc='upper left', handlelength=4)

# Set the title
ax_title = ax.set_title(f'{team1} passes vs {team2}', fontsize=30)

##############################################################################
# Plotting with grid.
# We will use mplsoccer's grid function to plot a pitch with a title and endnote axes.
fig, axs = pitch.grid(endnote_height=0.03, endnote_space=0, figheight=12,
                      title_height=0.06, title_space=0,
                      # Turn off the endnote/title axis. I usually do this after
                      # I am happy with the chart layout and text placement
                      axis=False,
                      grid_height=0.86)
fig.set_facecolor('#22312b')

# Plot the completed passes
lc1 = pitch.lines(df_pass[mask_complete].x, df_pass[mask_complete].y,
                  df_pass[mask_complete].end_x, df_pass[mask_complete].end_y,
                  lw=5, transparent=True, comet=True, label='completed passes',
                  color='#ad993c', ax=axs['pitch'])

# Plot the other passes
lc2 = pitch.lines(df_pass[~mask_complete].x, df_pass[~mask_complete].y,
                  df_pass[~mask_complete].end_x, df_pass[~mask_complete].end_y,
                  lw=5, transparent=True, comet=True, label='other passes',
                  color='#ba4f45', ax=axs['pitch'])

# fontmanager for google font (robotto)
robotto_regular = FontManager()

# setup the legend
legend = axs['pitch'].legend(facecolor='#22312b', handlelength=5, edgecolor='None',
                             prop=robotto_regular.prop, loc='upper left')
for text in legend.get_texts():
    text.set_fontsize(25)

# endnote and title
axs['endnote'].text(1, 0.5, '@your_twitter_handle', va='center', ha='right', fontsize=20,
                    fontproperties=robotto_regular.prop, color='#dee6ea')
ax_title = axs['title'].text(0.5, 0.5, f'{team1} passes vs {team2}', color='#dee6ea',
                             va='center', ha='center',
                             fontproperties=robotto_regular.prop, fontsize=25)

##############################################################################
# Filter datasets to only include passes leading to shots, and goals

TEAM1 = 'OL Reign'
TEAM2 = 'Houston Dash'
df_pass = df.loc[(df.pass_assisted_shot_id.notnull()) & (df.team_name == TEAM1),
                 ['x', 'y', 'end_x', 'end_y', 'pass_assisted_shot_id']]

df_shot = (df.loc[(df.type_name == 'Shot') & (df.team_name == TEAM1),
                  ['id', 'outcome_name', 'shot_statsbomb_xg']]
           .rename({'id': 'pass_assisted_shot_id'}, axis=1))

df_pass = df_pass.merge(df_shot, how='left').drop('pass_assisted_shot_id', axis=1)

mask_goal = df_pass.outcome_name == 'Goal'

##############################################################################
# This example shows how to plot all passes leading to shots from a team using a colormap (cmap).

# Setup the pitch
pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc',
                      half=True, pad_top=2)
fig, axs = pitch.grid(endnote_height=0.03, endnote_space=0, figheight=12,
                      title_height=0.08, title_space=0, axis=False,
                      grid_height=0.82)
fig.set_facecolor('#22312b')

# Plot the completed passes
pitch.lines(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y,
            lw=10, transparent=True, comet=True, cmap='jet',
            label='pass leading to shot', ax=axs['pitch'])

# Plot the goals
pitch.scatter(df_pass[mask_goal].end_x, df_pass[mask_goal].end_y, s=700,
              marker='football', edgecolors='black', c='white', zorder=2,
              label='goal', ax=axs['pitch'])
pitch.scatter(df_pass[~mask_goal].end_x, df_pass[~mask_goal].end_y,
              edgecolors='white', c='#22312b', s=700, zorder=2,
              label='shot', ax=axs['pitch'])

# endnote and title
axs['endnote'].text(1, 0.5, '@your_twitter_handle', va='center', ha='right', fontsize=25,
                    fontproperties=robotto_regular.prop, color='#dee6ea')
axs['title'].text(0.5, 0.5, f'{TEAM1} passes leading to shots \n vs {TEAM2}', color='#dee6ea',
                  va='center', ha='center',
                  fontproperties=robotto_regular.prop, fontsize=25)

# set legend
legend = axs['pitch'].legend(facecolor='#22312b', edgecolor='None',
                             loc='lower center', handlelength=4)
for text in legend.get_texts():
    text.set_fontproperties(robotto_regular.prop)
    text.set_fontsize(25)

plt.show()  # If you are using a Jupyter notebook you do not need this line

##############################################################################
# Alternative Theme + Shot contribution detection
# --------------
# Colors from the `Nord palette <https://www.nordtheme.com/>`_.

# @francescozonaro
rcParams['font.family'] = 'montserrat'
rcParams['text.color'] = 'black'

# Get event dataframe for game 7478
parser = Sbopen()
df, related, freeze, tactics = parser.event(7478)

##############################################################################
# Find the distance between the pass and the subsequent shot, checking
# that possession isn't lost.

df['distance_to_shot'] = 9999  # Initialize the new column

for idx, row in df.iterrows():
    if row['type_name'] == 'Pass':
        current_team = row['possession_team_name']
        current_team_name = row['team_name']
        
        next_shot_idx = idx + 1
        while next_shot_idx < len(df):
            next_event = df.iloc[next_shot_idx]
            if (next_event['type_name'] == 'Shot' and
                next_event['possession_team_name'] == current_team and
                next_event['team_name'] == current_team_name):
                distance = next_shot_idx - idx
                df.at[idx, 'distance_to_shot'] = distance
                break
            elif next_event['possession_team_name'] != current_team:
                break
            next_shot_idx += 1

df['distance_to_shot'] = 100/df['distance_to_shot']
old_min = df['distance_to_shot'].min()
old_max = df['distance_to_shot'].max()
new_min = 45
new_max = 200

df['distance_to_shot'] = ((df['distance_to_shot'] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

##############################################################################
# Boolean mask for filtering all the passes from a single player

team1, team2 = df.team_name.unique()
player = "Megan Anna Rapinoe"
mask_team1 = (df.type_name == 'Pass') & (df.team_name == team1) & (df.player_name == player)

##############################################################################
# Filter dataset to only include one player passes and get boolean mask for the completed passes

df_pass = df.loc[mask_team1, ['x', 'y', 'end_x', 'end_y', 'outcome_name', 'distance_to_shot']]
mask_complete = df_pass.outcome_name.isnull()
mask_shot_complete = (df_pass.outcome_name.isnull()) & (df_pass.distance_to_shot > new_min)

##############################################################################
# View the pass dataframe.
df_pass.head()

##############################################################################
# Plotting

# Set up the pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='#4C566A')
fig, ax = pitch.draw(figsize=(12, 10), constrained_layout=True, tight_layout=True)
fig.set_facecolor('white')

# Plot the completed passes
pitch.lines(df_pass[mask_complete].x, df_pass[mask_complete].y,
             df_pass[mask_complete].end_x, df_pass[mask_complete].end_y,
             lw=2, color='#A3BE8C', ax=ax)
pitch.scatter(df_pass[mask_shot_complete].end_x, df_pass[mask_shot_complete].end_y,
             s=df_pass[mask_shot_complete].distance_to_shot,edgecolor='black', color='#A3BE8C', ax=ax, zorder=9)
pitch.scatter(df_pass[~mask_shot_complete].end_x, df_pass[~mask_shot_complete].end_y,
             s=45,edgecolor='black', color='#A3BE8C', marker='X', ax=ax, zorder=9)


# Plot the other passes
pitch.lines(df_pass[~mask_complete].x, df_pass[~mask_complete].y,
             df_pass[~mask_complete].end_x, df_pass[~mask_complete].end_y,
             lw=2, color='#BF616A', ax=ax)
pitch.scatter(df_pass[~mask_complete].end_x, df_pass[~mask_complete].end_y,
             s=45, edgecolor='black', color='#BF616A', marker='X', ax=ax, zorder=9)

# Set the legend and the endnotes
legend_elements = [
        plt.scatter(
            [],
            [],
            s=55,
            edgecolor="black",
            linewidth=1,
            facecolor="#A3BE8C",
            zorder=7,
            marker="s",
            label="Completed pass",
        ),
        plt.scatter(
            [],
            [],
            s=55,
            edgecolor="black",
            linewidth=1,
            facecolor="#BF616A",
            zorder=7,
            marker="s",
            label="Other pass",
        ),
        plt.scatter(
            [],
            [],
            s=55,
            edgecolor="black",
            linewidth=1,
            facecolor="#ECEFF4",
            zorder=7,
            marker="o",
            label="Contribution to shot",
        ),
        plt.scatter(
            [],
            [],
            s=55,
            edgecolor="black",
            linewidth=1,
            facecolor="#ECEFF4",
            zorder=7,
            marker="X",
            label="No shot",
        ),
    ]

legend = ax.legend(
    facecolor="white",
    handles=legend_elements,
    loc="center",
    ncol=len(legend_elements),
    bbox_to_anchor=(0.5, 0.99),
    fontsize=15,
    fancybox=True,
    frameon=True,
    handletextpad=0.05,
    handleheight=1
)

ax.text(99.5, 82, "@your_twitter_handle", fontsize=12, va="center")
ax.text(
    0,
    82,
    f"{team1} vs {team2} ~ All {player} passes.",
    fontsize=11,
    va="center",
    ha="left",
)
ax.text(
    0,
    84,
    f"The quicker the shot happened after the pass, the bigger is the circle.",
    fontsize=11,
    va="center",
    ha="left",
)

plt.show()