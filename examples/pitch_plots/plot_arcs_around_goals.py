"""
==========================
Plotting arcs around goals
==========================

This example shows how to plot arcs around goals at specified distances on various pitch types.
"""
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt


def plot_arcs_around_goals_on_pitches(pitch_cls):
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.ravel()
    pitch_types = ['statsbomb', 'opta', 'tracab', 'skillcorner', 'wyscout',
                'metricasports', 'uefa', 'custom']

    placements = ('top', 'bottom') if pitch_cls == VerticalPitch else ('left', 'right')
    colors = ('red', 'blue')
    for idx, pt in enumerate(pitch_types):
        if pt in ['tracab', 'metricasports', 'custom', 'skillcorner']:
            pitch = pitch_cls(pitch_type=pt, pitch_length=105, pitch_width=68)
        else:
            pitch = pitch_cls(pitch_type=pt)
            
        pitch.draw(axes[idx])
        axes[idx].set_title(pt, fontsize=20, c='black', pad=15)
        for radius_meters in (5, 10, 20, 30):
            for color, placement in zip(colors, placements):
                pitch.add_arc_around_goal(axes[idx], radius_meters, placement=placement, color=color)
    return fig, axes


##############################################################################
# Horizontal pitches

plot_arcs_around_goals_on_pitches(Pitch)

##############################################################################
# Vertical pitches

plot_arcs_around_goals_on_pitches(VerticalPitch)