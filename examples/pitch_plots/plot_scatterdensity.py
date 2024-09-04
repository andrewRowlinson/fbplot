"""
===========
Scatter Density
===========

This example shows how to plot a scatter density plot based on a team's events.

Plot idea taken from https://github.com/LKremer/ggpointdensity as demonstrated by Eliot McKinley (
https://twitter.com/etmckinley/status/1169256582145703937) """

from mplsoccer.statsbomb import read_event, EVENT_SLUG
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# read data
df = read_event(f'{EVENT_SLUG}/7478.json',
                related_event_df=False, shot_freeze_frame_df=False,
                tactics_lineup_df=False)['event']

##############################################################################
# Filter ball receipts by Houston Dash
df = df[(df.team_name == 'Houston Dash') & (df.type_name == 'Ball Receipt')].copy()

##############################################################################
# Plotting

pitch = Pitch(pitch_color='lightgrey', line_color='white')
fig, ax = pitch.draw(figsize=(8, 6))
scatter = pitch.scatterdensity(df.x, df.y, ax=ax, sort=True, cmap='Purples', edgecolors='darkgrey', s=200)
fig.suptitle('Houston Dash Pass Receipts vs Seattle Reign')
plt.show()  # if you are not using a Jupyter notebook this is necessary to show the plot
