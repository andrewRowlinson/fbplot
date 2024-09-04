""" Module adds the plotting methods to the BasePitch abstract class."""

import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib import rcParams
from scipy.spatial import Voronoi, ConvexHull
from scipy.stats import circmean
from scipy.interpolate import interpn

from mplsoccer._pitch_base import BasePitch
from mplsoccer.heatmap import (bin_statistic, bin_statistic_positional,
                               bin_statistic_sonar, sonar, heatmap,
                               heatmap_positional)
from mplsoccer.linecollection import lines
from mplsoccer.quiver import arrows
from mplsoccer.scatterutils import scatter_football, scatter_rotation
from mplsoccer.utils import validate_ax, copy_doc


class BasePitchPlot(BasePitch):

    def plot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.plot,
        which automatically flips the x and y coordinates if the pitch is vertical.

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.plot.

        Returns
        -------
        lines : A list of Line2D objects representing the plotted data.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.plot([30, 35, 20], [30, 19, 40], ax=ax)
        """
        validate_ax(ax)
        x, y = self._reverse_if_vertical(x, y)
        return ax.plot(x, y, **kwargs)

    def scatter(self, x, y, rotation_degrees=None, marker=None, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.scatter,
        which automatically flips the x and y coordinates if the pitch is vertical.
        You can optionally use a football marker with marker='football' and rotate markers with
        rotation_degrees.

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        rotation_degrees: array-like or scalar, default None.
            Rotates the marker in degrees, clockwise. 0 degrees is facing the direction of play.
            In a horizontal pitch, 0 degrees is this way →, in a vertical pitch,
            0 degrees is this way ↑
        marker: MarkerStyle, optional
            The marker style. marker can be either an instance of the class or the
            text shorthand for a particular marker. Defaults to None, in which case it takes
            the value of rcParams["scatter.marker"] (default: 'o') = 'o'.
            If marker='football' plots a football shape with the pentagons the color
            of the edgecolors and hexagons the color of the 'c' argument; 'linewidths'
            also sets the linewidth of the football marker.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.scatter.

        Returns
        -------
        paths : matplotlib.collections.PathCollection
                or a tuple of (paths, paths) if marker='football'

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.scatter(30, 30, ax=ax)

        >>> from mplsoccer import Pitch
        >>> from mplsoccer import arrowhead_marker
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.scatter(30, 30, rotation_degrees=45, marker=arrowhead_marker, ax=ax)

        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.scatter(30, 30, marker='football', ax=ax)
        """
        validate_ax(ax)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)

        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        x, y = self._reverse_if_vertical(x, y)

        if marker is None:
            marker = rcParams['scatter.marker']

        if marker == 'football' and rotation_degrees is not None:
            raise NotImplementedError("rotated football markers are not implemented.")

        if marker == 'football':
            return scatter_football(x, y, ax=ax, **kwargs)
        if rotation_degrees is not None:
            return scatter_rotation(x, y, rotation_degrees, marker=marker,
                                    vertical=self.vertical, ax=ax, **kwargs)
        return ax.scatter(x, y, marker=marker, **kwargs)

    def _reflect_2d(self, x, y, standardized=False):
        """ Reflect data in the pitch lines."""
        x = np.ravel(x)
        y = np.ravel(y)
        if standardized:
            x_limits, y_limits = [0, 105], [0, 68]
        else:
            x_limits, y_limits = [self.dim.left, self.dim.right], [self.dim.bottom, self.dim.top]
        reflected_data_x = np.r_[x, 2 * x_limits[0] - x, 2 * x_limits[1] - x, x, x]
        reflected_data_y = np.r_[y, y, y, 2 * y_limits[0] - y, 2 * y_limits[1] - y]
        return reflected_data_x, reflected_data_y

    def kdeplot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around seaborn.kdeplot,
        which automatically flips the x and y coordinates
        if the pitch is vertical and clips to the pitch boundaries.

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to seaborn.kdeplot.

        Returns
        -------
        contour : matplotlib.contour.ContourSet

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(line_zorder=2)
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=100)
        >>> y = np.random.uniform(low=0, high=80, size=100)
        >>> pitch.kdeplot(x, y, cmap='Reds', fill=True, levels=100, ax=ax)
        """
        validate_ax(ax)

        x = np.ravel(x)
        y = np.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        x, y = self._reverse_if_vertical(x, y)

        return sns.kdeplot(x=x, y=y, ax=ax, clip=self.kde_clip, **kwargs)

    def hexbin(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.hexbin,
        which automatically flips the x and y coordinates if the pitch is vertical and
        clips to the pitch boundaries.

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        mincnt : int > 0, default: 1
            If not None, only display cells with more than mincnt number of points in the cell.
        gridsize : int or (int, int), default: (17, 8) for Pitch/ (17, 17) for VerticalPitch
            If a single int, the number of hexagons in the x-direction. The number of hexagons
            in the y-direction is chosen such that the hexagons are approximately regular.
            Alternatively, if a tuple (nx, ny), the number of hexagons in the x-direction
            and the y-direction.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.hexbin.

        Returns
        -------
        polycollection : matplotlib.collections.PolyCollection

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(line_zorder=2)
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=100)
        >>> y = np.random.uniform(low=0, high=80, size=100)
        >>> pitch.hexbin(x, y, edgecolors='black', gridsize=(11, 5), cmap='Reds', ax=ax)
        """
        validate_ax(ax)
        x = np.ravel(x)
        y = np.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        # according to seaborn hexbin isn't nan safe so filter out nan
        mask = np.isnan(x) | np.isnan(y)
        x = x[~mask]
        y = y[~mask]

        x, y = self._reverse_if_vertical(x, y)
        mincnt = kwargs.pop('mincnt', 1)
        gridsize = kwargs.pop('gridsize', self.hexbin_gridsize)
        extent = kwargs.pop('extent', self.hex_extent)
        hexbin = ax.hexbin(x, y, mincnt=mincnt, gridsize=gridsize, extent=extent, **kwargs)
        rect = patches.Rectangle((self.visible_pitch[0], self.visible_pitch[2]),
                                 self.visible_pitch[1] - self.visible_pitch[0],
                                 self.visible_pitch[3] - self.visible_pitch[2],
                                 fill=False)
        ax.add_patch(rect)
        hexbin.set_clip_path(rect)
        return hexbin

    def polygon(self, verts, ax=None, **kwargs):
        """ Plot polygons.
        Automatically flips the x and y vertices if the pitch is vertical.

        Parameters
        ----------
        verts: verts is a sequence of (verts0, verts1, ...)
            where verts_i is a numpy array of shape (number of vertices, 2).
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to
            matplotlib.patches.Polygon

        Returns
        -------
        list of matplotlib.patches.Polygon

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(label=True, axis=True)
        >>> fig, ax = pitch.draw()
        >>> shape1 = np.array([[50, 2], [80, 30], [40, 30], [40, 20]])
        >>> shape2 = np.array([[70, 70], [60, 50], [40, 40]])
        >>> verts = [shape1, shape2]
        >>> pitch.polygon(verts, color='red', alpha=0.3, ax=ax)
        """
        validate_ax(ax)
        patch_list = []
        for vert in verts:
            vert = np.asarray(vert)
            vert = self._reverse_vertices_if_vertical(vert)
            polygon = patches.Polygon(vert, closed=True, **kwargs)
            patch_list.append(polygon)
            ax.add_patch(polygon)
        return patch_list

    def goal_angle(self, x, y, ax=None, goal='right', **kwargs):
        """ Plot a polygon with the angle to the goal using matplotlib.patches.Polygon.
        See: https://matplotlib.org/stable/api/collections_api.html.
        Valid Collection keyword arguments: edgecolors, facecolors, linewidths, antialiaseds,
        transOffset, norm, cmap

        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.
        goal: str default 'right'.
            The goal to plot, either 'left' or 'right'.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to
             matplotlib.collections.PathCollection.

        Returns
        -------
        Polygon : matplotlib.patches.Polygon

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.goal_angle(100, 30, alpha=0.5, color='red', ax=ax)
        """
        validate_ax(ax)
        valid_goal = ['left', 'right']
        if goal not in valid_goal:
            raise TypeError(f'Invalid argument: goal should be in {valid_goal}')
        x = np.ravel(x)
        y = np.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        goal_coordinates = self.goal_right if goal == 'right' else self.goal_left
        verts = np.zeros((x.size, 3, 2))
        verts[:, 0, 0] = x
        verts[:, 0, 1] = y
        verts[:, 1:, :] = np.expand_dims(goal_coordinates, 0)
        return self.polygon(verts, ax=ax, **kwargs)

    def annotate(self, text, xy, xytext=None, ax=None, **kwargs):
        """ Utility wrapper around ax.annotate
        which automatically flips the xy and xytext coordinates if the pitch is vertical.

        Annotate the point xy with text.
        See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html

        Parameters
        ----------
        text : str
            The text of the annotation.
        xy : (float, float)
            The point (x, y) to annotate.
        xytext : (float, float), optional
            The position (x, y) to place the text at. If None, defaults to xy.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.annotate.

        Returns
        -------
        annotation : matplotlib.text.Annotation

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.annotate(text='center', xytext=(50, 50), xy=(60, 40), ha='center', va='center',
        ...                ax=ax, arrowprops=dict(facecolor='black'))
        """
        validate_ax(ax)
        xy = self._reverse_annotate_if_vertical(xy)
        if xytext is not None:
            xytext = self._reverse_annotate_if_vertical(xytext)
        return ax.annotate(text, xy, xytext, **kwargs)

    def text(self, x, y, s, ax=None, **kwargs):
        """ Utility wrapper around ax.text
        which automatically flips the x/y coordinates if the pitch is vertical.

        See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

        Parameters
        ----------
        x, y : float
            The position to place the text
        s : str
            The text
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.text.

        Returns
        -------
        text : matplotlib.text.Text

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.text(60, 40, 'Center of the pitch', va='center', ha='center', ax=ax)
        """
        validate_ax(ax)
        x, y = self._reverse_if_vertical(x, y)
        return ax.text(x, y, s, **kwargs)

    @copy_doc(bin_statistic)
    def bin_statistic(self, x, y, values=None, statistic='count', bins=(5, 4),
                      normalize=False, standardized=False):
        return bin_statistic(x, y, values=values, dim=self.dim, statistic=statistic,
                             bins=bins, normalize=normalize, standardized=standardized)

    @copy_doc(bin_statistic_sonar)
    def bin_statistic_sonar(self, x, y, angle, values=None,
                            statistic='count', bins=(5, 4, 10),
                            normalize=False, standardized=False, center=True):
        return bin_statistic_sonar(x, y, angle, values=values, dim=self.dim,
                                   statistic=statistic, bins=bins,
                                   normalize=normalize, standardized=standardized,
                                   center=center)

    @staticmethod
    @copy_doc(sonar)
    def sonar(stats_length, xindex=0, yindex=0,
              stats_color=None, cmap=None, vmin=None, vmax=None,
              rmin=0, rmax=None,
              sonar_alpha=1, sonar_facecolor='None',
              axis=False, label=False,
              ax=None,
              **kwargs):
        return sonar(stats_length, xindex=xindex, yindex=yindex,
                     stats_color=stats_color, cmap=cmap, vmin=vmin, vmax=vmax,
                     rmin=rmin, rmax=rmax,
                     sonar_alpha=sonar_alpha, sonar_facecolor=sonar_facecolor,
                     axis=axis, label=label, ax=ax, **kwargs)

    def sonar_grid(self, stats_length,
                   stats_color=None, cmap=None, vmin=None, vmax=None,
                   rmin=0, rmax=None,
                   sonar_alpha=1, sonar_facecolor='None',
                   axis=False, label=False,
                   width=None, height=None,
                   exclude_zeros=True, exclude_nan=True,
                   ax=None, **kwargs):
        """ Plot a grid of polar bar charts on an existing axes.

        Parameters
        ----------
        stats_length : dict
            This should be calculated via bin_statistic_sonar().
            It controls the length of the bars.
        stats_color : dict, default None
            This should be calculated via bin_statistic_sonar().
            It controls the color of the bars via a cmap. The vmin/vmax
            arguments will set the boundaries for the cmap.
            If stats_color is None then the color of the bars is controlled
            by 'color', 'fc', or 'facecolor' arguments.
        cmap : str or matplotlib.colros.Colormap, default None
            Controls the color of the bars via stats_color.
        vmin, vmax : float, default None
            The cmap is mapped linearly to the range vmin to vmax, so that values
            equal to or less than vmin are given the first color in the cmap
            and values equal to or greater than vmax are given the last color
            in the cmap. The default of None sets the values to the minimum value of
            stats_color['statistic'] and the maximum value of stats_color['statistic'].
        rmin, rmax : float, default 0 and None
            The radial axis limits. The default rmax of None sets the values to the maximum
            of stats_length['statistic'].
        sonar_alpha : float, default 1
            The alpha/ transparency of the sonar axes patch.
        sonar_facecolor : any Matplotlib color, default 'None'
            The facecolor of the sonar axes. The default 'None' makes the axes transparent.
        axis : bool, default False
            Whether to set the axis spines to visible.
        label : bool, default False
            Whether to include the axis labels.
        width, height : float, default None
            The width, height of the inset Polar axes in the x/y data coordinates.
            You should only provide one of the width or height arguments
            since the Polar axes are square and the other values is set dynamically.
        exclude_zeros : bool, default True
            Whether to draw the Polar axes where all the values are zero for the grid cell.
        exclude_nan : bool, default True
            Whether to draw the Polar axes where all the values are numpy.nan for the grid cell.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.bar.

        Examples
        --------
        >>> from mplsoccer import Pitch, Sbopen
        >>> parser = Sbopen()
        >>> df = parser.event(69251)[0]
        >>> df = df[(df.type_name == 'Pass') &
        ...         (df.outcome_name.isnull())].copy()
        >>> pitch = Pitch()
        >>> angle, distance = pitch.calculate_angle_and_distance(df.x, df.y,
        ...                                                      df.end_x, df.end_y)
        >>> bs = pitch.bin_statistic_sonar(df.x, df.y, angle,
        ...                                bins=(6, 4, 4), center=True)
        >>> fig, ax = pitch.draw(figsize=(8, 5.5))
        >>> axs = pitch.sonar_grid(bs, width=10, fc='cornflowerblue',
        ...                        ec='black', ax=ax)
        """
        validate_ax(ax)
        mask_zero = np.all(np.isclose(stats_length['statistic'], 0), axis=2)
        mask_null = np.all(np.isnan(stats_length['statistic']), axis=2)
        axs = np.empty(stats_length['cx'].shape, dtype='O')
        it = np.nditer(stats_length['cx'], flags=['multi_index'])
        for cx in it:
            if mask_zero[it.multi_index] and exclude_zeros:
                ax_inset = None
            elif mask_null[it.multi_index] and exclude_nan:
                ax_inset = None
            else:
                ax_inset = self.inset_axes(cx, stats_length['cy'][it.multi_index],
                                        width=width, height=height, ax=ax, polar=True)
                sonar(stats_length=stats_length,
                    xindex=it.multi_index[1], yindex=it.multi_index[0],
                    stats_color=stats_color, cmap=cmap, vmin=vmin, vmax=vmax,
                    rmin=rmin, rmax=rmax,
                    sonar_alpha=sonar_alpha, sonar_facecolor=sonar_facecolor,
                    axis=axis, label=label,
                    ax=ax_inset, **kwargs)
            axs[it.multi_index] = ax_inset
        axs = np.squeeze(axs)
        if axs.size == 1:
            axs = axs.item()
        return axs

    @copy_doc(heatmap)
    def heatmap(self, stats, ax=None, **kwargs):
        return heatmap(stats, ax=ax, vertical=self.vertical, **kwargs)

    @copy_doc(bin_statistic_positional)
    def bin_statistic_positional(self, x, y, values=None, positional='full',
                                 statistic='count', normalize=False):
        return bin_statistic_positional(x, y, values=values,
                                        dim=self.dim, positional=positional,
                                        statistic=statistic, normalize=normalize)

    @copy_doc(heatmap_positional)
    def heatmap_positional(self, stats, ax=None, **kwargs):
        return heatmap_positional(stats, ax=ax, vertical=self.vertical, **kwargs)

    def label_heatmap(self, stats, str_format=None, exclude_zeros=False, exclude_nan=False,
                      xoffset=0, yoffset=0, ax=None, **kwargs):
        """ Labels the heatmap(s) and automatically flips the coordinates if the pitch is vertical.

        Parameters
        ----------
        stats : A dictionary or list of dictionaries.
            This should be calculated via bin_statistic_positional() or bin_statistic().
        str_format : str
            A format string passed to str_format.format() to format the labels.
        exclude_zeros : bool, default False
            Whether to exclude zeros when labelling the heatmap.
        exclude_nan : bool, default False
            Whether to exclude numpy.nan when labelling the heatmap.
        xoffset, yoffset : float, default 0
            The amount in data coordinates to offset the labels from the center of the grid cell.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.

        **kwargs : All other keyword arguments are passed on to matplotlib.text.Text.

        Returns
        -------
        text : A list of matplotlib.text.Text.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> import matplotlib.patheffects as path_effects
        >>> pitch = Pitch(line_zorder=2, pitch_color='black')
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=100)
        >>> y = np.random.uniform(low=0, high=80, size=100)
        >>> stats = pitch.bin_statistic(x, y)
        >>> pitch.heatmap(stats, edgecolors='black', cmap='hot', ax=ax)
        >>> path_eff = [path_effects.Stroke(linewidth=0.5, foreground='#22312b')]
        >>> text = pitch.label_heatmap(stats, color='white', ax=ax, fontsize=20, ha='center',
        ...                            va='center', path_effects=path_eff, str_format='{:.0f}')
        """
        validate_ax(ax)
        va = kwargs.pop('va', 'center')
        ha = kwargs.pop('ha', 'center')

        if not isinstance(stats, list):
            stats = [stats]

        annotation_list = []
        for bin_stat in stats:
            # remove labels outside the plot extents
            mask_x_outside1 = bin_stat['cx'] < self.dim.pitch_extent[0]
            mask_x_outside2 = bin_stat['cx'] > self.dim.pitch_extent[1]
            mask_y_outside1 = bin_stat['cy'] < self.dim.pitch_extent[2]
            mask_y_outside2 = bin_stat['cy'] > self.dim.pitch_extent[3]
            mask_clip = mask_x_outside1 | mask_x_outside2 | mask_y_outside1 | mask_y_outside2
            if exclude_zeros:
                mask_clip = mask_clip | (np.isclose(bin_stat['statistic'], 0.))
            if exclude_nan:
                mask_clip = mask_clip | np.isnan(bin_stat['statistic'])
            mask_clip = np.ravel(mask_clip)

            text = np.ravel(bin_stat['statistic'])[~mask_clip]
            cx = np.ravel(bin_stat['cx'])[~mask_clip] + xoffset
            cy = np.ravel(bin_stat['cy'])[~mask_clip] + yoffset
            for idx, text_str in enumerate(text):
                if str_format is not None:
                    text_str = str_format.format(text_str)
                annotation = self.text(cx[idx], cy[idx], text_str, ax=ax,
                                       va=va, ha=ha, **kwargs)
                annotation_list.append(annotation)

        return annotation_list

    @copy_doc(arrows)
    def arrows(self, xstart, ystart, xend, yend, *args, ax=None, **kwargs):
        validate_ax(ax)
        return arrows(xstart, ystart, xend, yend, *args, ax=ax, vertical=self.vertical, **kwargs)

    @copy_doc(lines)
    def lines(self, xstart, ystart, xend, yend, color=None, n_segments=100,
              comet=False, transparent=False, alpha_start=0.01,
              alpha_end=1, cmap=None, ax=None, **kwargs):
        validate_ax(ax)
        return lines(xstart, ystart, xend, yend, color=color, n_segments=n_segments,
                     comet=comet, transparent=transparent, alpha_start=alpha_start,
                     alpha_end=alpha_end, cmap=cmap, ax=ax, vertical=self.vertical,
                     reverse_cmap=self.reverse_cmap, **kwargs)

    def convexhull(self, x, y):
        """ Get lines of Convex Hull for a set of coordinates

        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.

        Returns
        -------
        hull_vertices: a numpy array of vertoces [1, num_vertices, [x, y]] of the Convex Hull.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=11)
        >>> y = np.random.uniform(low=0, high=80, size=11)
        >>> hull = pitch.convexhull(x, y)
        >>> poly = pitch.polygon(hull, ax=ax, facecolor='cornflowerblue', alpha=0.3)
        """
        points = np.vstack([x, y]).T
        hull = ConvexHull(points)
        return points[hull.vertices].reshape(1, -1, 2)

    def scatterdensity(self, x, y, ax, cmap='viridis', sort=True, **kwargs):
        """ Generate scatter density plot - a scatter plot where point colour represents 2D density at that point.
        Inspired by https://github.com/LKremer/ggpointdensity

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.

        ax : matplotlib.axes.Axes, default None
            The axis to plot on.

        cmap : str or matplotlib Colormap, default 'viridis'
             A Colormap instance or registered colormap name to use for density colouring

        sort : boolean, default True
             Sort the points so the highest densities are plotted last and are thus on top

        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.scatter.

        Returns
        -------
        paths : matplotlib.collections.PathCollection
                or a tuple of (paths, paths) if marker='football'

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=1500)
        >>> y = np.random.uniform(low=0, high=80, size=1500)
        >>> scatter = pitch.scatterdensity(x,y, ax=ax, cmap='magma')
        """
        # Need to convert x,y to numpy array if not already
        x = np.array(x)
        y = np.array(y)

        # The following implementation is taken from https://stackoverflow.com/a/53865762/3015186 by Guillaume
        # It is a lot faster than the in-built scipy kde method whilst retaining visual quality overlaid on a pitch
        data, x_e, y_e = np.histogram2d(x, y, bins=20, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        scatter = self.scatter(x, y, ax=ax, c=z, cmap=cmap, **kwargs)
        return scatter

    def voronoi(self, x, y, teams):
        """ Get Voronoi vertices for a set of coordinates.
        Uses a trick by Dan Nichol (@D4N__ on Twitter) where points are reflected in the pitch lines
        before calculating the Voronoi. This means that the Voronoi extends to
        the edges of the pitch. See:
        https://github.com/ProformAnalytics/tutorial_nbs/blob/master/notebooks/Voronoi%20Reflection%20Trick.ipynb

        Players outside the pitch dimensions are assumed to be standing on the pitch edge.
        This means that their coordinates are clipped to the pitch edges
        before calculating the Voronoi.

        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.

        teams: array-like or scalar.
            This splits the results into the Voronoi vertices for each team.
            This can either have integer (1/0) values or boolean (True/False) values.
            team1 is where team==1 or team==True
            team2 is where team==0 or team==False

        Returns
        -------
        team1 : a 1d numpy array (length number of players in team 1) of 2d arrays
            Where the individual 2d arrays are coordinates of the Voronoi vertices.

        team2 : a 1d numpy array (length number of players in team 2) of 2d arrays
            Where the individual 2d arrays are coordinates of the Voronoi vertices.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=0, high=120, size=22)
        >>> y = np.random.uniform(low=0, high=80, size=22)
        >>> teams = np.array([0] * 11 + [1] * 11)
        >>> pitch.scatter(x[teams == 0], y[teams == 0], color='red', ax=ax)
        >>> pitch.scatter(x[teams == 1], y[teams == 1], color='blue', ax=ax)
        >>> team1, team2 = pitch.voronoi(x, y, teams)
        >>> team1_poly = pitch.polygon(team1, ax=ax, color='blue', alpha=0.3)
        >>> team2_poly = pitch.polygon(team2, ax=ax, color='red', alpha=0.3)
        """
        x = np.ravel(x)
        y = np.ravel(y)
        teams = np.ravel(teams)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        if teams.size != x.size:
            raise ValueError("x and team must be the same size")

        if not self.dim.aspect_equal:
            standardized = True
            x, y = self.standardizer.transform(x, y)
            extent = np.array([0, 105, 0, 68])
        else:
            standardized = False
            extent = self.dim.pitch_extent

        # clip outside to pitch extents
        x = x.clip(min=extent[0], max=extent[1])
        y = y.clip(min=extent[2], max=extent[3])

        # reflect in pitch lines
        reflect_x, reflect_y = self._reflect_2d(x, y, standardized=standardized)
        reflect = np.vstack([reflect_x, reflect_y]).T

        # create Voronoi
        vor = Voronoi(reflect)

        # get region vertices
        regions = vor.point_region[:x.size]
        regions = np.array(vor.regions, dtype='object')[regions]
        region_vertices = []
        for region in regions:
            verts = vor.vertices[region]
            verts[:, 0] = np.clip(verts[:, 0], a_min=extent[0], a_max=extent[1])
            verts[:, 1] = np.clip(verts[:, 1], a_min=extent[2], a_max=extent[3])
            # convert coordinates back if previously standardized
            if standardized:
                x_std, y_std = self.standardizer.transform(verts[:, 0], verts[:, 1], reverse=True)
                verts[:, 0] = x_std
                verts[:, 1] = y_std
            region_vertices.append(verts)
        region_vertices = np.array(region_vertices, dtype='object')

        # seperate team1/ team2 vertices
        team1 = region_vertices[teams == 1]
        team2 = region_vertices[teams == 0]

        return team1, team2

    def calculate_angle_and_distance(self, xstart, ystart, xend, yend, degrees=False):
        """ Calculates the angle in radians counter-clockwise and the distance
        between a start and end location. Where the angle 0 is this way →
        (the straight line from left to right) in a horizontally orientated pitch
        and this way ↑ in a vertically orientated pitch.
        The angle goes from 0 to 2pi. To convert the angle to degrees clockwise use degrees=True.

        Parameters
        ----------
        xstart, ystart, xend, yend: array-like or scalar.
            Commonly, these parameters are 1D arrays.
            These should be the start and end coordinates to calculate the angle between.
        degrees : bool, default False
            If False, the angle is returned in radians counter-clockwise in the range [0, 2pi]
            If True, the angle is returned in degrees clockwise in the range [0, 360].

        Returns
        -------
        angle: ndarray
            The default is an array of angles in radians counter-clockwise in the range [0, 2pi].
            Where 0 is the straight line left to right in a horizontally orientated pitch
            and the straight line bottom to top in a vertically orientated pitch.
            If degrees = True, then the angle is returned in degrees clockwise in the range [0, 360]
        distance: ndarray
            Array of distances.

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> pitch.calculate_angle_and_distance(0, 40, 30, 20, degrees=True)
        (array([326.30993247]), array([36.05551275]))
        """
        xstart = np.ravel(xstart)
        ystart = np.ravel(ystart)
        xend = np.ravel(xend)
        yend = np.ravel(yend)
        if xstart.size != ystart.size:
            raise ValueError("xstart and ystart must be the same size")
        if xstart.size != xend.size:
            raise ValueError("xstart and xend must be the same size")
        if ystart.size != yend.size:
            raise ValueError("ystart and yend must be the same size")

        if not self.dim.aspect_equal:
            xstart, ystart = self.standardizer.transform(xstart, ystart)
            xend, yend = self.standardizer.transform(xend, yend)
            standardized = True
        else:
            standardized = False

        x_dist = xend - xstart
        if self.dim.invert_y and standardized is False:
            y_dist = ystart - yend
        else:
            y_dist = yend - ystart

        angle = np.arctan2(y_dist, x_dist)
        # if negative angle make positive angle, so goes from 0 to 2 * pi
        angle[angle < 0] = 2 * np.pi + angle[angle < 0]

        if degrees:
            # here we convert to degrees and take the negative for clockwise angles
            # the modulus is not strictly necessary for plotting purposes,
            # but gives the postive angle in degrees
            angle = np.mod(-np.degrees(angle), 360)

        distance = (x_dist ** 2 + y_dist ** 2) ** 0.5

        return angle, distance

    def flow(self, xstart, ystart, xend, yend, bins=(5, 4), arrow_type='same', arrow_length=5,
             color=None, ax=None, **kwargs):
        """ Create a flow map by binning the data into cells and calculating the average
        angles and distances.

        Parameters
        ----------
        xstart, ystart, xend, yend: array-like or scalar.
            Commonly, these parameters are 1D arrays.
            These should be the start and end coordinates to calculate the angle between.
        bins : int or [int, int] or array_like or [array, array], optional
            The bin specification for binning the data to calculate the angles/ distances.
              * the number of bins for the two dimensions (nx = ny = bins),
              * the number of bins in each dimension (nx, ny = bins),
              * the bin edges for the two dimensions (x_edge = y_edge = bins),
              * the bin edges in each dimension (x_edge, y_edge = bins).
                If the bin edges are specified, the number of bins will be,
                (nx = len(x_edge)-1, ny = len(y_edge)-1).
        arrow_type : str, default 'same'
            The supported arrow types are: 'same', 'scale', and 'average'.
            'same' makes the arrows the same size (arrow_length).
            'scale' scales the arrow length by the average distance
            in the cell (up to a max of arrow_length).
            'average' makes the arrow size the average distance in the cell.
        arrow_length : float, default 5
            The arrow_length for the flow map. If the arrow_type='same',
            all the arrows will be arrow_length. If the arrow_type='scale',
            the arrows will be scaled by the average distance.
            If the arrow_type='average', the arrows_length is ignored
            This is automatically multipled by 100 if using a 'tracab' pitch
            (i.e. the default is 500).
        color : A matplotlib color, defaults to None.
            Defaults to None. In that case the marker color is
            determined by the cmap (default 'viridis').
            and the counts of the starting positions in each bin.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.quiver.

        Returns
        -------
        PolyCollection : matplotlib.quiver.Quiver

        Examples
        --------
        >>> from mplsoccer import Pitch, Sbopen
        >>> parser = Sbopen()
        >>> df, related, freeze, tactics = parser.event(7478)
        >>> team1, team2 = df.team_name.unique()
        >>> mask_team1 = (df.type_name == 'Pass') & (df.team_name == team1)
        >>> df = df[mask_team1].copy()
        >>> pitch = Pitch(line_zorder=2)
        >>> fig, ax = pitch.draw()
        >>> bs_heatmap = pitch.bin_statistic(df.x, df.y, statistic='count', bins=(6, 4))
        >>> hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
        >>> fm = pitch.flow(df.x, df.y, df.end_x, df.end_y, color='black', arrow_type='same',
        ...                 arrow_length=6, bins=(6, 4), headwidth=2, headlength=2,
        ...                 headaxislength=2, ax=ax)
        """
        validate_ax(ax)
        # calculate  the binned statistics
        angle, distance = self.calculate_angle_and_distance(xstart, ystart, xend, yend)

        if not self.dim.aspect_equal:
            standardized = True
            # slightly inefficient as we also transform the data
            # in the calculate_angle_and_distance method
            # but I wanted to make it easier for users as this way they do not need
            # to know whether their data needs
            # to be transformed for calculate_angle_and_distance
            xstart, ystart = self.standardizer.transform(xstart, ystart)
            xend, yend = self.standardizer.transform(xend, yend)
        else:
            standardized = False

        bs_distance = self.bin_statistic(xstart, ystart, values=distance,
                                         statistic='mean', bins=bins, standardized=standardized)
        bs_angle = self.bin_statistic(xstart, ystart, values=angle,
                                      statistic=circmean, bins=bins, standardized=standardized)

        # calculate the arrow length
        if self.dim.pad_multiplier != 1:
            arrow_length = arrow_length * self.dim.pad_multiplier
        if arrow_type == 'scale':
            new_d = (bs_distance['statistic'] * arrow_length /
                     np.nan_to_num(bs_distance['statistic']).max(initial=None))
        elif arrow_type == 'same':
            new_d = arrow_length
        elif arrow_type == 'average':
            new_d = bs_distance['statistic']
        else:
            valid_arrows = ['scale', 'same', 'average']
            raise TypeError(f'Invalid argument: arrow_type should be in {valid_arrows}')

        # calculate the end positions of the arrows
        endx = bs_angle['cx'] + (np.cos(bs_angle['statistic']) * new_d)
        if self.dim.invert_y and not standardized:
            endy = bs_angle['cy'] - (np.sin(bs_angle['statistic']) * new_d)  # invert_y
        else:
            endy = bs_angle['cy'] + (np.sin(bs_angle['statistic']) * new_d)

        # get coordinates and convert back to the pitch coordinates if necessary
        cx, cy = bs_angle['cx'], bs_angle['cy']
        if standardized:
            cx, cy = self.standardizer.transform(cx, cy, reverse=True)
            endx, endy = self.standardizer.transform(endx, endy, reverse=True)

        # plot arrows
        if color is not None:
            return self.arrows(cx, cy, endx, endy, color=color, ax=ax, **kwargs)
        bs_count = self.bin_statistic(xstart, ystart, statistic='count',
                                      bins=bins, standardized=standardized)
        return self.arrows(cx, cy, endx, endy, bs_count['statistic'], ax=ax, **kwargs)

    def triplot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.triplot

        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.triplot.
        """
        validate_ax(ax)
        x, y = self._reverse_if_vertical(x, y)
        return ax.triplot(x, y, **kwargs)

    # The methods below for drawing/ setting attributes for some pitch elements
    # are defined in pitch.py (Pitch/ VerticalPitch classes)
    # as they differ for horizontal/ vertical pitches
    def _scale_pad(self):
        pass

    def _set_aspect(self):
        pass

    def _set_extent(self):
        pass

    def _validate_pad(self):
        pass

    def _draw_rectangle(self, ax, x, y, width, height, **kwargs):
        pass

    def _draw_centered_rectangle(self, ax, x, y, width, height, **kwargs):
        pass

    def _draw_line(self, ax, x, y, **kwargs):
        pass

    def _draw_ellipse(self, ax, x, y, width, height, **kwargs):
        pass

    def _draw_arc(self, ax, x, y, width, height, theta1, theta2, **kwargs):
        pass

    def _draw_stripe(self, ax, i):
        pass

    def _draw_stripe_grass(self, pitch_color):
        pass

    @staticmethod
    def _reverse_if_vertical(x, y):
        pass

    @staticmethod
    def _reverse_vertices_if_vertical(vert):
        pass

    @staticmethod
    def _reverse_annotate_if_vertical(annotate):
        pass
