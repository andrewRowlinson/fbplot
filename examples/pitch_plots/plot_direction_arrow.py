"""
=========
Direction play/attack arrow
=========

This example shows how to add a direction of play to pitch plotting in mplsoccer
example of this is https://twitter.com/Worville/status/1365738445826633733?s=20`._
"""
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


pitch = Pitch(pad_left=20, pad_right=20, pad_bottom=20, pad_top=20)
# specifying figure size (width, height)
fig, ax = pitch.draw(figsize=(10, 5))
arrow = mpatches.FancyArrowPatch((40, 85), (80, 85),
                                 mutation_scale=25)
ax.add_patch(arrow)
pitch.draw(ax=ax)


verticalPitch = VerticalPitch(pad_left=20, pad_right=20)
fig, ax1 = verticalPitch.draw(figsize=(4, 8))
# specifying figure size (width, height)
arrow = mpatches.FancyArrowPatch((83.3, 45), (83.3, 85),
                                 mutation_scale=25)
ax1.add_patch(arrow)
verticalPitch.draw(ax=ax1)

plt.show()
