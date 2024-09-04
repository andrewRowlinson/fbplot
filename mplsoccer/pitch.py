""" Module containing: Pitch and VerticalPitch, which are used to plot pitches in mplsoccer"""

import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from mplsoccer._pitch_plot import BasePitchPlot

__all__ = ['Pitch', 'VerticalPitch']


class Pitch(BasePitchPlot):

    def _set_aspect(self):
        self.aspect = self.dim.aspect

    def _scale_pad(self):
        self.pad_left = self.pad_left * self.aspect
        self.pad_right = self.pad_right * self.aspect

    def _validate_pad(self):
        # make sure padding not too large for the pitch
        if abs(min(self.pad_left, 0) + min(self.pad_right, 0)) >= self.dim.length:
            raise ValueError("pad_left/pad_right too negative for pitch length")
        if abs(min(self.pad_top, 0) + min(self.pad_bottom, 0)) >= self.dim.width:
            raise ValueError("pad_top/pad_bottom too negative for pitch width")
        if self.half:
            if abs(min(self.pad_left, 0) + min(self.pad_right, 0)) >= self.dim.length / 2:
                raise ValueError("pad_left/pad_right too negative for pitch length")

    def _set_extent(self):
        extent = np.array([self.dim.left, self.dim.right,
                           self.dim.bottom, self.dim.top], dtype=np.float32)
        pad = np.array([-self.pad_left, self.pad_right,
                        -self.pad_bottom, self.pad_top], dtype=np.float32)
        visible_pad = np.clip(np.array([self.pad_left, self.pad_right,
                                        self.pad_bottom, self.pad_top],
                                       dtype=np.float32), a_min=None, a_max=0.)
        visible_pad[[0, 2]] = - visible_pad[[0, 2]]
        if self.half:
            extent[0] = self.dim.center_length  # pitch starts at center line
            visible_pad[0] = - self.pad_left  # do not want clipped values if half
        if self.dim.invert_y:  # when inverted the padding is negative
            pad[2:] = -pad[2:]
            visible_pad[2:] = - visible_pad[2:]
        self.extent = extent + pad
        self.ax_aspect = (abs(self.extent[1] - self.extent[0]) /
                          (abs(self.extent[3] - self.extent[2]) * self.aspect))
        self.visible_pitch = extent + visible_pad
        if self.half:
            extent[0] = extent[0] - min(self.pad_left, self.dim.pitch_length/2)

        # hexbin
        self.hexbin_gridsize = (17, 8)
        self.hex_extent = np.array([self.dim.left, self.dim.right,
                                    min(self.dim.bottom, self.dim.top),
                                    max(self.dim.bottom, self.dim.top)], dtype=np.float32)

        # kdeplot
        self.kde_clip = ((self.dim.left, self.dim.right), (self.dim.bottom, self.dim.top))

        # lines
        self.reverse_cmap = self.dim.invert_y

        # vertical for lines/ arrows
        self.vertical = False

        # stripe
        total_height = abs(self.extent[3] - self.extent[2])
        pad_top, pad_bottom = -min(self.pad_top, 0), min(self.pad_bottom, 0)
        if self.dim.invert_y:
            pad_top, pad_bottom = -pad_top, -pad_bottom
        top_side = abs(self.extent[2] - self.dim.top + pad_top)
        bottom_side = abs(self.extent[2] - self.dim.bottom + pad_bottom)
        self.stripe_end = top_side / total_height
        self.stripe_start = bottom_side / total_height
        self.grass_stripe_end = int((1 - self.stripe_start) * 1000)
        self.grass_stripe_start = int((1 - self.stripe_end) * 1000)

    def _draw_rectangle(self, ax, x, y, width, height, **kwargs):
        if self.dim.invert_y:
            height = - height
        rectangle = patches.Rectangle((x, y), width, height, **kwargs)
        ax.add_patch(rectangle)
        return rectangle

    def _draw_centered_rectangle(self, ax, x, y, width, height, **kwargs):
        if self.dim.invert_y:
            height = - height
        x = x - width / 2
        y = y - height / 2
        rectangle = patches.Rectangle((x, y), width, height, **kwargs)
        ax.add_patch(rectangle)
        return rectangle

    def _draw_line(self, ax, x, y, **kwargs):
        line = Line2D(x, y, **kwargs)
        ax.add_artist(line)

    def _draw_ellipse(self, ax, x, y, width, height, **kwargs):
        ellipse = patches.Ellipse((x, y), width, height, **kwargs)
        ax.add_patch(ellipse)

    def _draw_arc(self, ax, x, y, width, height, theta1, theta2, **kwargs):
        arc = patches.Arc((x, y), width, height, theta1=theta1, theta2=theta2, **kwargs)
        ax.add_patch(arc)

    def _draw_stripe(self, ax, i):
        ax.axvspan(self.dim.stripe_locations[i], self.dim.stripe_locations[i + 1],  # note axvspan
                   self.stripe_start, self.stripe_end,
                   facecolor=self.stripe_color, zorder=self.stripe_zorder)

    def _draw_stripe_grass(self, pitch_color):
        total_width = self.extent[1] - self.extent[0]
        for i in range(len(self.dim.stripe_locations) - 1):
            if i % 2 == 0:
                if ((self.extent[0] <= self.dim.stripe_locations[i] <= self.extent[1]) or
                        (self.extent[0] <= self.dim.stripe_locations[i + 1] <= self.extent[1])):
                    start = (int((max(self.dim.stripe_locations[i], self.extent[0]) -
                                  self.extent[0]) / total_width * 1000))
                    end = (int((min(self.dim.stripe_locations[i+1], self.extent[1]) -
                                self.extent[0]) / total_width * 1000))
                    pitch_color[self.grass_stripe_start: self.grass_stripe_end, start: end] = \
                        pitch_color[self.grass_stripe_start: self.grass_stripe_end, start: end] + 2
        return pitch_color

    @staticmethod
    def _reverse_if_vertical(x, y):
        return x, y

    @staticmethod
    def _reverse_vertices_if_vertical(vert):
        return vert

    @staticmethod
    def _reverse_annotate_if_vertical(annotate):
        return annotate

    def add_arc_around_goal(self, ax, radius_meters, placement='left', **kwargs):
        """Add an arc to the pitch centred on one of the goals.
        
        Parameters
        ----------
        ax : matplotlib axis
            A matplotlib.axes.Axes to draw the arc on.
        radius_meters : float
            The radius of the arc in meters.
        placement : str, default 'left'
            Which side of the pitch to plot the arc. Should be either 'left' or 'right'.
        kwargs : dict
            Additional keyword arguments accepted by matplotlib.patches.Arc.

        Returns
        -------
            Arc added to `ax` and None returned.

        Raises
        ------
        ValueError
            If `placement` is not 'left' or 'right'.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.add_arc_around_goal(ax, radius_meters=10, placement="left")
        >>> pitch.add_arc_around_goal(ax, radius_meters=20, placement="right", color="red")
        """
        xmin, xmax, ymin, ymax = self.extent
        length_pixels = abs(xmax - xmin)
        width_pixels = abs(ymax - ymin)
        
        # scale down via ratio of radius to dimension
        if self.pitch_type != 'metricasports':
            width = 2 * length_pixels * radius_meters / self.dim.length
            height =  2 * width_pixels * radius_meters / self.dim.width / self.dim.aspect
            if self.pitch_type == 'tracab':
                # has unit centimeters
                width *= 100
                height *= 100
        elif self.pitch_type == 'metricasports':
            width = 2 * length_pixels * radius_meters / self.dim.pitch_length
            height = 2 * width_pixels * radius_meters / self.dim.pitch_width 
            
        if self.dim.origin_center:
            y = 0
        else:
            y = width_pixels / 2 - self.pad_bottom
        if placement == 'left':
            self._draw_arc(
                ax, xmin + self.pad_left, y, width, height, theta1=-90, theta2=90, **kwargs
            )
        elif placement == 'right':
            self._draw_arc(
                ax, xmax - self.pad_right, y, width, height, theta1=90, theta2=270, **kwargs
            )
        else:
            raise ValueError(
                f'placement={placement} should be "left" or "right"'
            )
    

class VerticalPitch(BasePitchPlot):

    def _set_aspect(self):
        self.aspect = 1 / self.dim.aspect

    def _scale_pad(self):
        self.pad_bottom = self.pad_bottom / self.aspect
        self.pad_top = self.pad_top / self.aspect

    def _validate_pad(self):
        # make sure padding not too large for the pitch
        if abs(min(self.pad_bottom, 0) + min(self.pad_top, 0)) >= self.dim.length:
            raise ValueError("pad_bottom/pad_top too negative for pitch length")
        if abs(min(self.pad_left, 0) + min(self.pad_right, 0)) >= self.dim.width:
            raise ValueError("pad_left/pad_right too negative for pitch width")
        if self.half:
            if abs(min(self.pad_bottom, 0) + min(self.pad_top, 0)) >= self.dim.length / 2:
                raise ValueError("pad_bottom/pad_top too negative for pitch length")

    def _set_extent(self):
        extent = np.array([self.dim.top, self.dim.bottom,
                           self.dim.left, self.dim.right], dtype=np.float32)
        pad = np.array([self.pad_left, -self.pad_right, -self.pad_bottom,
                        self.pad_top], dtype=np.float32)
        visible_pad = np.clip(np.array([self.pad_left, self.pad_right,
                                        self.pad_bottom, self.pad_top], dtype=np.float32),
                              a_min=None, a_max=0.)
        visible_pad[[1, 2]] = - visible_pad[[1, 2]]
        if self.half:
            extent[2] = self.dim.center_length  # pitch starts at center line
            visible_pad[2] = - self.pad_bottom  # do not want clipped values if half
        if self.dim.invert_y:  # when inverted the padding is negative
            pad[0:2] = -pad[0:2]
            visible_pad[0:2] = - visible_pad[0:2]
        self.extent = extent + pad
        self.ax_aspect = (abs(self.extent[1] - self.extent[0]) /
                          (abs(self.extent[3] - self.extent[2]) * self.aspect))
        self.visible_pitch = extent + visible_pad
        if self.half:
            extent[2] = extent[2] - min(self.pad_bottom, self.dim.pitch_length/2)

        # hexbin
        self.hexbin_gridsize = (17, 17)
        self.hex_extent = np.array([min(self.dim.bottom, self.dim.top),
                                    max(self.dim.bottom, self.dim.top),
                                    self.dim.left, self.dim.right], dtype=np.float32)

        # kdeplot
        self.kde_clip = ((self.dim.top, self.dim.bottom), (self.dim.left, self.dim.right))

        # lines
        self.reverse_cmap = False

        # vertical for lines/ arrows
        self.vertical = True

        # stripe
        total_height = abs(self.extent[1] - self.extent[0])
        pad_top, pad_bottom = -min(self.pad_left, 0), min(self.pad_right, 0)
        if self.dim.invert_y:
            pad_top, pad_bottom = -pad_top, -pad_bottom
        top_side = abs(self.extent[0] - self.dim.top + pad_top)
        bottom_side = abs(self.extent[0] - self.dim.bottom + pad_bottom)
        self.stripe_start = top_side / total_height
        self.stripe_end = bottom_side / total_height
        self.grass_stripe_end = int(self.stripe_end * 1000)
        self.grass_stripe_start = int(self.stripe_start * 1000)

    def _draw_rectangle(self, ax, x, y, width, height, **kwargs):
        if self.dim.invert_y:
            height = - height
        rectangle = patches.Rectangle((y, x), height, width, **kwargs)
        ax.add_patch(rectangle)
        return rectangle

    def _draw_centered_rectangle(self, ax, x, y, width, height, **kwargs):
        if self.dim.invert_y:
            height = - height
        x = x - width / 2
        y = y - height / 2
        rectangle = patches.Rectangle((y, x), height, width, **kwargs)
        ax.add_patch(rectangle)
        return rectangle

    def _draw_line(self, ax, x, y, **kwargs):
        line = Line2D(y, x, **kwargs)
        ax.add_artist(line)

    def _draw_ellipse(self, ax, x, y, width, height, **kwargs):
        ellipse = patches.Ellipse((y, x), height, width, **kwargs)
        ax.add_patch(ellipse)

    def _draw_arc(self, ax, x, y, width, height, theta1, theta2, **kwargs):
        arc = patches.Arc((y, x), height, width, theta1=theta1 + 90, theta2=theta2 + 90, **kwargs)
        ax.add_patch(arc)

    def _draw_stripe(self, ax, i):
        ax.axhspan(self.dim.stripe_locations[i], self.dim.stripe_locations[i + 1],  # note axhspan
                   self.stripe_start, self.stripe_end,
                   facecolor=self.stripe_color, zorder=self.stripe_zorder)

    def _draw_stripe_grass(self, pitch_color):
        total_width = self.extent[3] - self.extent[2]
        for i in range(len(self.dim.stripe_locations) - 1):
            if i % 2 == 0:
                if ((self.extent[2] <= self.dim.stripe_locations[i] <= self.extent[3]) or
                        (self.extent[2] <= self.dim.stripe_locations[i + 1] <= self.extent[3])):
                    start = (1000 - int((min(self.dim.stripe_locations[i+1],
                                             self.extent[3]) - self.extent[2])
                                        / total_width * 1000))
                    end = (1000 - int((max(self.dim.stripe_locations[i],
                                           self.extent[2]) - self.extent[2])
                                      / total_width * 1000))
                    pitch_color[start: end, self.grass_stripe_start: self.grass_stripe_end] = \
                        pitch_color[start: end, self.grass_stripe_start: self.grass_stripe_end] + 2
        return pitch_color

    @staticmethod
    def _reverse_if_vertical(x, y):
        return y, x

    @staticmethod
    def _reverse_vertices_if_vertical(vert):
        return vert[:, [1, 0]].copy()

    @staticmethod
    def _reverse_annotate_if_vertical(annotate):
        return annotate[::-1]
    
    def add_arc_around_goal(self, ax, radius_meters, placement, **kwargs):
        """Add an arc to the pitch centred on one of the goals.
        
        Parameters
        ----------
        ax : matplotlib axis
            A matplotlib.axes.Axes to draw the arc on.
        radius_meters : float
            The radius of the arc in meters.
        placement : str, default 'top'
            Which side of the pitch to plot the arc. Should be either 'top' or 'bottom'.
        kwargs : dict
            Additional keyword arguments accepted by matplotlib.patches.Arc.

        Returns
        -------
            Arc added to `ax` and None returned.

        Raises
        ------
        ValueError
            If `placement` is not 'top' or 'bottom'.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.add_arc_around_goal(ax, radius_meters=15, placement="bottom")
        >>> pitch.add_arc_around_goal(ax, radius_meters=7, placement="top", color="blue")
        """
        xmin, xmax, ymin, ymax = self.extent
        width_pixels = abs(xmax - xmin)
        length_pixels = abs(ymax - ymin)
        
        # scale down via ratio of radius to dimension
        if self.pitch_type != "metricasports":
            width = 2 * width_pixels * self.dim.aspect * radius_meters / self.dim.width
            height = 2 * length_pixels * radius_meters / self.dim.length
            if self.pitch_type == "tracab":
                # has unit centimeters
                width *= 100
                height *= 100
        elif self.pitch_type == "metricasports":
            width = 2 * width_pixels * radius_meters / self.dim.pitch_width
            height = 2 * length_pixels * radius_meters / self.dim.pitch_length
            
        if self.dim.origin_center:
            x = 0
        else:
            x = width_pixels / 2 - self.pad_left
        if placement == 'bottom':
            self._draw_arc(
                ax, ymin + self.pad_bottom, x, height, width, theta1=-90, theta2=90, **kwargs
            )
        elif placement == 'top':
            self._draw_arc(
                ax, ymax - self.pad_top, x, height, width, theta1=90, theta2=270, **kwargs
            )
        else:
            raise ValueError(
                f'placement={placement} should be "top" or "bottom"'
            )
